import time
import os
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from tqdm.auto import trange
import matplotlib.pyplot as plt

from custom_types import (
    XArray, XCovariance, PRNGKeyArray, 
    Float, Array, SDEType, 
    SampleType
)
from configs import get_blob_config, get_gmm_config, save_config
from rf import (
    RectifiedFlow, ResidualNetwork, 
    get_timestep_embedding, get_flow_soln_kwargs, 
    velocity_to_score, score_to_velocity,
    get_rectified_flow,
    cosine_time,
    identity
)
from train import make_step, apply_ema, test_train
from sample import (
    get_x_sampler, get_x_y_sampler, get_non_singular_sample_fn, single_sample_fn_ode 
)
from utils import (
    get_opt_and_state, clear_and_get_results_dir, 
    create_gif, plot_losses, plot_samples, 
    clip_latents, get_data, measurement,
    exists, default, maybe_clip, maybe_invert,
    plot_losses_ppca
)
from ppca import run_ppca
from soap import soap


if __name__ == "__main__":
    key = jr.key(int(time.time()))

    config = get_blob_config()

    save_dir = clear_and_get_results_dir(save_dir="imgs_{}/".format(config.data.dataset))

    save_config(config, filename=os.path.join(save_dir, "config.yml"))

    print("Running on {} dataset.".format(config.data.dataset))

    key_net, key_data, key_measurement, key_ppca, key_em = jr.split(key, 5)

    # Rectified flow model and EMA
    flow = get_rectified_flow(
        config.data.data_dim, 
        config.model.width_size, 
        config.model.depth, 
        config.model.time_embedding_dim, 
        config.model.soln_kwargs, 
        key=key_net
    )

    if config.train.use_ema:
        ema_flow = deepcopy(flow)

    # Optimiser
    opt, opt_state = get_opt_and_state(
        flow, 
        config.train.optimiser,
        lr=config.train.lr, 
        use_lr_schedule=config.train.use_lr_schedule, 
        initial_lr=config.train.initial_lr,
        n_epochs_warmup=config.train.n_epochs_warmup,
        n_data=config.data.n_data,
        n_batch=config.train.n_batch,
        diffusion_iterations=config.train.diffusion_iterations
    )

    # Latents
    X = get_data(key_data, config.data.n_data, dataset=config.data.dataset)

    # Generate y ~ G[y|x, cov_y]
    cov_y = jnp.eye(config.data.data_dim) * jnp.square(config.data.sigma_y)
    keys = jr.split(key_measurement, config.data.n_data)
    Y = jax.vmap(partial(measurement, cov_y=cov_y))(keys, X) 

    if config.train.test_on_latents:
        test_train(
            key, 
            flow, 
            X, 
            n_batch=config.train.n_batch,
            diffusion_iterations=config.train.diffusion_iterations, 
            use_ema=config.train.use_ema,
            sde_type=config.train.sde_type,
            save_dir=save_dir
        )

    # PPCA pre-training for q_0(x|mu_x, cov_x)
    if config.train.ppca_pretrain:
        mu_x, cov_x, X_Y = run_ppca(
            key_ppca, 
            flow,
            Y, 
            cov_y=cov_y,
            n_pca_iterations=config.train.n_pca_iterations, 
            sde_type=config.train.sde_type, 
            sampling_mode=config.train.sampling_mode,
            X=X
        )
    else:
        X_Y = jax.vmap(partial(measurement, cov_y=5. * cov_y))(keys, X) # Testing

    # Initial model samples
    sampler = get_x_sampler(
        flow, 
        sampling_mode=config.train.sampling_mode, 
        sde_type=config.train.sde_type
    )
    X_test = jax.vmap(sampler)(keys)

    # Plot initial samples
    plot_samples(X, X_Y, Y, X_test, save_dir=save_dir)

    # Expectation maximisation
    losses_k = []
    for k in range(config.train.em_iterations):
        key_k, key_sample = jr.split(jr.fold_in(key_em, k))

        # Train on sampled latents
        losses_s = []
        with trange(
            config.train.diffusion_iterations, desc="Training", colour="green"
        ) as steps:
            for s in steps:
                key_x, key_step = jr.split(jr.fold_in(key_k, s))

                xy = jr.choice(key_x, X_Y, (config.train.n_batch,)) # Make sure always choosing x ~ p(x|y)

                L, flow, key, opt_state = make_step(
                    flow, 
                    xy, 
                    key_step, 
                    opt_state, 
                    opt.update, 
                    loss_type=config.train.loss_type, 
                    time_schedule=config.train.time_schedule
                )

                if config.train.use_ema:
                    ema_flow = apply_ema(ema_flow, flow, config.train.ema_rate)

                losses_s.append(L)
                steps.set_postfix_str("k={:04d} L={:.3E}".format(k, L))

        # Plot losses
        losses_k += losses_s
        plot_losses(
            losses_k, k, config.train.diffusion_iterations, save_dir=save_dir
        )

        # Generate latents from q(x|y)
        sampler = get_x_y_sampler(
            ema_flow if config.train.use_ema else flow, 
            cov_y=cov_y, 
            mu_x=mu_x, 
            cov_x=cov_x, 
            sde_type=config.train.sde_type,
            sampling_mode=config.train.sampling_mode, 
            q_0_sampling=True
        )
        keys = jr.split(key_sample, config.data.n_data)
        X_Y = jax.vmap(sampler)(keys, Y)

        if config.train.clip_x_y:
            X_Y = clip_latents(X_Y, config.train.x_clip_limit) 

        # Generate latents from q(x)
        sampler = get_x_sampler(
            ema_flow if config.train.use_ema else flow, 
            sampling_mode=config.train.sampling_mode, 
            sde_type=config.train.sde_type
        )
        X_test = jax.vmap(sampler)(keys)

        # Plot latents
        plot_samples(X, X_Y, Y, X_test, iteration=k + 1, save_dir=save_dir)

        if k > 1:
            create_gif(save_dir)

        # Restart optimiser on previously trained score network
        if config.train.re_init_opt_state:
            opt, opt_state = get_opt_and_state(
                flow, 
                config.train.optimiser,
                lr=config.train.lr, 
                use_lr_schedule=config.train.use_lr_schedule, 
                initial_lr=config.train.initial_lr,
                n_epochs_warmup=config.train.n_epochs_warmup,
                n_data=config.data.n_data,
                n_batch=config.train.n_batch,
                diffusion_iterations=config.train.diffusion_iterations
            )