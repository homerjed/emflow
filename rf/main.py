import time
import os
from copy import deepcopy
from functools import partial
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker
from tqdm.auto import trange
import matplotlib.pyplot as plt

# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

from rf import (
    RectifiedFlow, ResidualNetwork, 
    get_timestep_embedding, get_flow_soln_kwargs, 
    velocity_to_score, score_to_velocity
)
from train import make_step, apply_ema, cosine_time, identity
from sample import get_x_sampler, get_x_y_sampler
from utils import (
    ppca, get_opt_and_state, clear_and_get_results_dir, 
    create_gif, plot_losses, plot_samples, 
    clip_latents, get_data, measurement,
    exists, default, maybe_clip, maybe_invert,
    plot_losses_ppca, gaussian_log_prob
)
from soap import soap


def test_train(key, flow, X, n_batch, diffusion_iterations, use_ema, save_dir):
    # Test whether training config fits FM model on latents

    opt, opt_state = get_opt_and_state(flow, soap, lr=1e-3)

    if use_ema:
        ema_flow = deepcopy(flow)

    losses_i = []
    with trange(
        diffusion_iterations, desc="Training (test)", colour="red"
    ) as steps:
        for i in steps:
            key_x, key_step = jr.split(jr.fold_in(key, i))

            x = jr.choice(key_x, X, (n_batch,)) # Make sure always choosing x ~ p(x|y)

            L, flow, key, opt_state = make_step(
                flow, 
                x, 
                key_step, 
                opt_state, 
                opt.update, 
                loss_type="mse", 
                time_schedule=identity
            )

            if use_ema:
                ema_flow = apply_ema(ema_flow, flow, ema_rate)

            losses_i.append(L)
            steps.set_postfix_str("L={:.3E}".format(L))

    key, key_sample = jr.split(key)

    # Generate latents from q(x)
    keys = jr.split(key_sample, 8000)
    X_test_sde = jax.vmap(get_non_singular_sample_fn(flow))(keys)
    X_test_ode = jax.vmap(partial(single_sample_fn_ode, flow, sde_type=sde_type))(keys)

    plt.figure()
    plt.scatter(*X.T, s=0.1, color="k")
    plt.scatter(*X_test_sde.T, s=0.1, color="b")
    plt.scatter(*X_test_ode.T, s=0.1, color="r")
    plt.savefig(os.path.join(save_dir, "test.png"))
    plt.close()


if __name__ == "__main__":
    key = jr.key(int(time.time()))

    save_dir             = clear_and_get_results_dir(save_dir="imgs/")

    # Train
    test_on_latents      = False
    em_iterations        = 64
    diffusion_iterations = 5_000
    n_batch              = 5_000
    loss_type            = "mse"
    time_schedule        = identity
    lr                   = 1e-3
    optimiser            = soap
    use_lr_schedule      = False 
    initial_lr           = 1e-6
    n_epochs_warmup      = 1
    ppca_pretrain        = True
    n_pca_iterations     = 32
    clip_x_y             = False # Clip sampled latents
    x_clip_limit         = 4.
    re_init_opt_state    = True
    sampling_mode        = "ode"        # ODE, SDE or DDIM
    sde_type             = "zero-ends"  # SDE of flow ODE
    mode                 = "full"       # CG mode or not
    use_ema              = True
    ema_rate             = 0.999

    # Model
    width_size           = 256 
    depth                = 3 
    activation           = jax.nn.gelu 
    soln_kwargs          = dict(t0=0., dt0=0.02, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea
    time_embedding_dim   = 64

    # Data
    data_dim             = 2
    n_data               = 100_000
    sigma_y              = 0.02 # Tiny eigenvalues may have been numerically unstable?
    cov_y                = jnp.eye(data_dim) * jnp.square(sigma_y)

    assert sampling_mode in ["ddim", "ode", "sde"]
    assert sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert mode in ["full", "cg"]

    key_net, key_data, key_measurement, key_ppca, key_em = jr.split(key, 5)

    # Rectified flow model
    time_embedder = get_timestep_embedding(time_embedding_dim)

    net = ResidualNetwork(
        data_dim, 
        width_size=width_size, 
        depth=depth, 
        t_embedding_dim=time_embedding_dim, 
        t1=soln_kwargs["t1"],
        key=key_net 
    )

    flow = RectifiedFlow(
        net, time_embedder, x_shape=(data_dim,), **soln_kwargs
    )

    if use_ema:
        ema_flow = deepcopy(flow)

    # Optimiser
    opt, opt_state = get_opt_and_state(
        flow, 
        optimiser,
        lr=lr, 
        use_lr_schedule=use_lr_schedule, 
        initial_lr=initial_lr,
        n_epochs_warmup=n_epochs_warmup,
        n_data=n_data,
        n_batch=n_batch,
        diffusion_iterations=diffusion_iterations
    )

    # Latents
    X = get_data(key_data, n_data)

    # Generate y ~ G[y|x, cov_y]
    keys = jr.split(key_measurement, n_data)
    Y = jax.vmap(partial(measurement, cov_y=cov_y))(keys, X) 

    if test_on_latents:
        test_train(
            key, 
            flow, 
            X, 
            n_batch=n_batch,
            diffusion_iterations=diffusion_iterations, 
            use_ema=use_ema,
            save_dir=save_dir
        )

    # PPCA pre-training for q_0(x|mu_x, cov_x)
    mu_x = jnp.zeros(data_dim)
    cov_x = jnp.identity(data_dim) # Start at cov_y?
    if ppca_pretrain:

        X_ = Y

        log_probs_x = []
        with trange(
            n_pca_iterations, desc="PPCA Training", colour="blue"
        ) as steps:
            for s in steps:                
                key_pca, key_sample = jr.split(jr.fold_in(key_ppca, s))

                sampler = get_x_y_sampler(
                    flow, 
                    cov_y=cov_y, 
                    mu_x=mu_x, 
                    cov_x=cov_x, 
                    sde_type=sde_type,
                    sampling_mode=sampling_mode, 
                    q_0_sampling=True
                )
                keys = jr.split(key_sample, n_data)
                X_ = jax.vmap(sampler)(keys, Y)

                mu_x, cov_x = ppca(X_, key_pca, rank=data_dim)

                l_x = -jnp.mean(jax.vmap(partial(gaussian_log_prob, mu_x=mu_x, cov_x=cov_x))(X))
                log_probs_x.append(l_x)

                steps.set_postfix_str("L_x={:.3E}".format(l_x))

        print("mu/cov x: {} \n {}".format(mu_x, cov_x))

        plot_losses_ppca(log_probs_x)
    else:
        X_ = jax.vmap(partial(measurement, cov_y=5. * cov_y))(keys, X) # Testing

    # Initial model samples
    sampler = get_x_sampler(
        flow, sampling_mode=sampling_mode, sde_type=sde_type
    )
    X_test = jax.vmap(sampler)(keys)

    # Plot initial samples
    plot_samples(X, X_, Y, X_test, save_dir=save_dir)

    # Expectation maximisation
    losses_k = []
    for k in range(em_iterations):
        key_k, key_sample = jr.split(jr.fold_in(key_em, k))

        # Train on sampled latents
        losses_i = []
        with trange(
            diffusion_iterations, desc="Training", colour="green"
        ) as steps:
            for i in steps:
                key_x, key_step = jr.split(jr.fold_in(key_k, i))

                x = jr.choice(key_x, X_, (n_batch,)) # Make sure always choosing x ~ p(x|y)

                L, flow, key, opt_state = make_step(
                    flow, 
                    x, 
                    key_step, 
                    opt_state, 
                    opt.update, 
                    loss_type=loss_type, 
                    time_schedule=time_schedule
                )

                if use_ema:
                    ema_flow = apply_ema(ema_flow, flow, ema_rate)

                losses_i.append(L)
                steps.set_postfix_str("k={:04d} L={:.3E}".format(k, L))

        # Plot losses
        losses_k += losses_i
        plot_losses(losses_k, k, diffusion_iterations, save_dir=save_dir)

        # Generate latents from q(x|y)
        sampler = get_x_y_sampler(
            ema_flow if use_ema else flow, 
            cov_y=cov_y, 
            mu_x=mu_x, 
            cov_x=cov_x, 
            sde_type=sde_type,
            sampling_mode=sampling_mode, 
            q_0_sampling=True
        )
        keys = jr.split(key_sample, n_data)
        X_ = jax.vmap(sampler)(keys, Y)

        if clip_x_y:
            X_ = clip_latents(X_, x_clip_limit) 

        # Generate latents from q(x)
        sampler = get_x_sampler(
            ema_flow if use_ema else flow, 
            sampling_mode=sampling_mode, 
            sde_type=sde_type
        )
        X_test = jax.vmap(sampler)(keys)

        # Plot latents
        plot_samples(X, X_, Y, X_test, iteration=k + 1, save_dir=save_dir)

        if k > 1:
            create_gif(save_dir)

        # Restart optimiser on previously trained score network
        if re_init_opt_state:
            opt, opt_state = get_opt_and_state(
                flow, 
                optimiser,
                lr=lr, 
                use_lr_schedule=use_lr_schedule, 
                initial_lr=initial_lr,
                n_epochs_warmup=n_epochs_warmup,
                n_data=n_data,
                n_batch=n_batch,
                diffusion_iterations=diffusion_iterations
            )