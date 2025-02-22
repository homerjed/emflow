import time
import os
from copy import deepcopy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from einops import rearrange
from tqdm.auto import trange
import matplotlib.pyplot as plt

from custom_types import (
    XArray, XCovariance, PRNGKeyArray, 
    Float, Array, SDEType, 
    SampleType
)
from configs import (
    get_blob_config, get_gmm_config, get_mnist_config, save_config
)
from rf import (
    RectifiedFlow, ResidualNetwork, 
    get_timestep_embedding, get_flow_soln_kwargs, 
    velocity_to_score, score_to_velocity,
    get_rectified_flow,
    cosine_time,
    identity
)
from train import make_step, apply_ema
from sample import (
    get_x_sampler, get_x_y_sampler, 
    get_non_singular_sample_fn, single_sample_fn_ode 
)
from utils import (
    get_opt_and_state, clear_and_get_results_dir, 
    create_gif, plot_losses, plot_samples, 
    clip_latents, plot_losses_ppca,
    flatten, unflatten
)
from soap import soap


def test_train(key, flow, X, n_batch, diffusion_iterations, use_ema, ema_rate, sde_type, save_dir):
    # Test whether training config fits FM model on latents

    opt, opt_state = get_opt_and_state(flow, soap, lr=1e-3)

    if use_ema:
        ema_flow = deepcopy(flow)

    key_steps, key_sample = jr.split(key)

    losses_s = []
    with trange(
        diffusion_iterations, desc="Training (test)", colour="red"
    ) as steps:
        for s, x in zip(
            steps, get_loader(X, key=key_steps).loop(n_batch)
        ):
            key_x, key_step = jr.split(jr.fold_in(key_steps, s))

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

            losses_s.append(L)
            steps.set_postfix_str("L={:.3E}".format(L))

    # Generate latents from q(x)
    n_sample = 16
    keys = jr.split(key_sample, n_sample)
    X_test_sde = jax.vmap(get_non_singular_sample_fn(flow))(keys)
    X_test_ode = jax.vmap(partial(single_sample_fn_ode, flow, sde_type=sde_type))(keys)

    X_test_sde = rearrange(X, "(r c) 1 h w -> (r h) (c w)", r=4, c=4)
    X_test_ode = rearrange(X, "(r c) 1 h w -> (r h) (c w)", r=4, c=4)

    fig, axs = plt.subplots(1, 3, figsize=(15., 5.))
    ax = axs[0]
    ax.imshow(X.T, s=0.1, color="k")
    ax = axs[1]
    ax.imshow(X_test_ode)
    ax = axs[2]
    ax.imshow(X_test_sde.T)
    for ax in axs:
        ax.axis("off")
    plt.savefig(os.path.join(save_dir, "test.png"), bbox_inches="tight")
    plt.close()


# @typecheck
@eqx.filter_jit
def ppca(
    x: Float[Array, "n d"], 
    key: PRNGKeyArray, 
    rank: int 
) -> tuple[XArray, XCovariance]:
    # Probabilistic PCA

    samples, features = x.shape

    mu_x = jnp.mean(x, axis=0)
    x = x - mu_x

    if samples < features:
        C = x @ x.T / samples
    else:
        C = x.T @ x / samples # Sample covariance

    if rank < len(C) // 5:
        Q = jr.normal(key, (len(C), rank))
        L, Q, _ = jax.experimental.sparse.linalg.lobpcg_standard(C, Q)
    else:
        L, Q = jnp.linalg.eigh(C)
        L, Q = L[-rank:], Q[:, -rank:]

    if samples < features:
        Q = x.T @ Q
        Q = Q / jnp.linalg.norm(Q, axis=0)

    if rank < features:
        D = (jnp.trace(C) - jnp.sum(L)) / (features - rank)
    else:
        D = jnp.asarray(1e-6)

    # U = Q * jnp.sqrt(jnp.maximum(L - D, 0.0))

    # cov_x = jnp.cov(D, rowvar=False)
    # cov_x = DPLR(D * jnp.ones(features), U, U.T)

    return mu_x, C #jnp.eye(features) * D #C # cov_x

def run_ppca(
    key: PRNGKeyArray, 
    flow: RectifiedFlow,
    Y, # NOTE: type
    cov_y: Float[Array, "d d"],
    sde_type: SDEType, 
    sampling_mode: SampleType,
    n_pca_iterations: int = 256,
    X: Optional[Float[Array, "n d"]] = None, # Only for loss, debugging
) -> tuple[XArray, XCovariance, Float[Array, "n d"]]:

    X_Y = flatten(Y)

    n_data, data_dim = X_Y.shape

    mu_x = jnp.zeros(data_dim)
    cov_x = jnp.identity(data_dim) # Start at cov_y?

    # Batched PPCA to estimate mu_x, cov_x
    log_probs_x = []
    with trange(
        n_pca_iterations, desc="PPCA Training", colour="blue"
    ) as steps:
        for s, x_y in zip(steps, get_loader(X_Y, key=key)):                
            key_pca, key_sample = jr.split(jr.fold_in(key, s))

            # x_y = jax.vmap(partial(measurement, cov_y=cov_y))(x_y)

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
            x_y = jax.vmap(sampler)(keys, x_y)

            mu_x, cov_x = ppca(flatten(x_y), key_pca, rank=data_dim)

            if X is not None:
                log_likelihood = partial(gaussian_log_prob, mu_x=mu_x, cov_x=cov_x)
                l_x = -jnp.mean(jax.vmap(log_likelihood)(X))
                log_probs_x.append(l_x)

            steps.set_postfix_str("L_x={:.3E}".format(l_x))

    print("mu/cov x: {} \n {}".format(mu_x, cov_x))

    if X is not None:
        plot_losses_ppca(log_probs_x)

    X_Y = unflatten(X_Y, *Y.shape[2:])

    return mu_x, cov_x, X_Y



if __name__ == "__main__":
    from utils import mnist, get_loader

    def measurement(
        key: PRNGKeyArray, 
        x: Float[Array, "_ _ _"], 
        cov_y: Float[Array, "d d"]
    ) -> Float[Array, "d"]: 
        # Sample from G[y|x, cov_y]
        y = jr.multivariate_normal(key, flatten(x), cov_y) 
        return unflatten(y, *x.shape[2:])

    key = jr.key(int(time.time()))

    config = get_mnist_config() 

    # Latents
    # trainloader, _ = mnist(key, config.model.img_size)
    X = mnist(key, config.model.img_size, return_arrays=True)

    save_dir = clear_and_get_results_dir(save_dir="imgs_{}/".format(config.data.dataset))

    save_config(config, filename=os.path.join(save_dir, "config.yml"))

    print("Running on {} dataset.".format(config.data.dataset))

    key_net, key_data, key_measurement, key_ppca, key_em = jr.split(key, 5)

    # Rectified flow model and EMA
    flow = get_rectified_flow(config.model, key=key_net)

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

    # Generate y ~ G[y|x, cov_y] on the fly
    cov_y = jnp.identity(config.data.data_dim) * jnp.square(config.data.sigma_y)
    keys = jr.split(key_measurement, X.shape[0])
    Y = jax.vmap(partial(measurement, cov_y=cov_y))(keys, X) 

    if config.train.test_on_latents:
        test_train(
            key, 
            flow, 
            X,
            n_batch=config.train.n_batch,
            diffusion_iterations=config.train.diffusion_iterations, 
            use_ema=config.train.use_ema,
            ema_rate=config.train.ema_rate,
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
            X=None
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
        key_k, key_sample, key_loader = jr.split(jr.fold_in(key_em, k), 3)

        # Train on sampled latents
        losses_s = []
        with trange(
            config.train.diffusion_iterations, desc="Training", colour="green"
        ) as steps:
            for s, xy in zip(steps, get_loader(X_Y, key=key_loader)):
                key_x, key_step = jr.split(jr.fold_in(key_k, s))

                # xy = jr.choice(key_x, X_Y, (config.train.n_batch,)) # Make sure always choosing x ~ p(x|y)

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