import time
import os
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm.auto import trange

from configs import get_mnist_config, save_config
from rf import get_rectified_flow
from train import make_step, apply_ema, test_train
from sample import get_x_sampler, get_x_y_sampler
from utils import (
    get_opt_and_state, clear_and_get_results_dir, 
    create_gif, plot_losses, plot_samples, 
    clip_latents, get_data, measurement,
    get_loader, get_shardings, maybe_shard, 
    exists, flatten, unflatten
)
from ppca import run_ppca

from typing import NamedTuple

class DataShapes(NamedTuple):
    # Reshape to these where required
    x_array: int
    y_array: int
    x_image: tuple[int]
    y_image: tuple[int]


if __name__ == "__main__":

    key = jr.key(int(time.time()))

    key_net, key_data, key_measurement, key_ppca, key_em = jr.split(key, 5)

    replicated_sharding, distributed_sharding = get_shardings()

    config = get_mnist_config()

    save_dir = clear_and_get_results_dir(
        save_dir="out/imgs_{}/".format(config.data.dataset)
    )

    save_config(config, filename=os.path.join(save_dir, "config.yml"))

    print("Running on {} dataset.".format(config.data.dataset))

    # Rectified flow model and EMA
    flow = get_rectified_flow(config.model, key=key_net)
    flow = maybe_shard(flow, replicated_sharding)

    if config.train.use_ema:
        ema_flow = deepcopy(flow)
        ema_flow = maybe_shard(ema_flow, replicated_sharding)

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

    # Latents x ~ p(x)
    X, A = get_data(key_data, config)

    # Generate y ~ G[y|A @ x, cov_y] NOTE: is A the same or different for every observation?
    keys = jr.split(key_measurement, config.data.n_data)
    cov_y = jnp.identity(config.data.data_dim) * jnp.square(config.data.sigma_y)

    if exists(A):
        # NOTE: flatten only for mnist! preprocess_fn that flattens if so required?
        Y = jax.vmap(lambda key, x, A: measurement(key, flatten(x), A=A, cov_y=cov_y))(keys, X, A) 
    else:
        Y = jax.vmap(lambda key, x: measurement(key, x, A=None, cov_y=cov_y))(keys, X)

    # Flatten after each sampling; unflatten to plot only
    # > reshaping to image shape in DiT 
    postprocess_fn = lambda x: flatten(x) 

    print("X, Y", X.shape, Y.shape)

    # Test if config can fit to p(x)
    if config.train.test_on_latents:
        test_train(
            key, 
            flow, 
            X, 
            n_batch=config.train.n_batch,
            diffusion_iterations=config.train.diffusion_iterations, 
            lr=config.train.lr,
            use_ema=config.train.use_ema,
            ema_rate=config.train.ema_rate,
            sde_type=config.train.sde_type,
            postprocess_fn=postprocess_fn,
            img_size=config.data.img_size,
            replicated_sharding=replicated_sharding,
            distributed_sharding=distributed_sharding,
            save_dir=save_dir
        )

    # PPCA pre-training for q_0(x|mu_x, cov_x) NOTE: can batch sample like below with a larger dataset than that used for PPCA...
    if config.train.ppca_pretrain:
        mu_x, cov_x, X_Y = run_ppca(
            key_ppca, 
            flow,
            config.data.latent_dim,
            Y, 
            A, 
            cov_y=cov_y,
            n_pca_iterations=config.train.n_pca_iterations, 
            sde_type=config.train.sde_type, 
            sampling_mode=config.train.sampling_mode,
            mode=config.train.mode,
            max_steps=config.train.max_steps,
            postprocess_fn=postprocess_fn,
            replicated_sharding=replicated_sharding,
            distributed_sharding=distributed_sharding,
            X=X,
            save_dir=save_dir
        )
    else:
        X_Y = jax.vmap(partial(measurement, cov_y=5. * cov_y))(keys, X, A) # Testing

    # Initial model samples
    sampler = get_x_sampler(
        flow, 
        sampling_mode=config.train.sampling_mode, 
        sde_type=config.train.sde_type,
        postprocess_fn=postprocess_fn
    )
    X_ = jax.vmap(sampler)(keys)

    # Plot initial samples
    plot_samples(X, X_Y, Y, X_, save_dir=save_dir)

    # Expectation maximisation
    losses_k = []
    for k in range(config.train.em_iterations):
        key_k, key_sample = jr.split(jr.fold_in(key_em, k))

        loader = get_loader(
            maybe_shard(unflatten(X_Y, config.data.img_shape), distributed_sharding), 
            key=key_k
        )

        # Train on sampled latents
        losses_s = []
        with trange(
            config.train.diffusion_iterations, desc="Training", colour="green"
        ) as steps:
            for s, xy in zip(
                steps, 
                loader.loop(config.train.n_batch)
            ):
                key_x, key_step = jr.split(jr.fold_in(key_k, s))

                L, flow, key, opt_state = make_step(
                    flow, 
                    xy, 
                    key_step, 
                    opt_state, 
                    opt.update, 
                    loss_type=config.train.loss_type, 
                    time_schedule=config.train.time_schedule,
                    replicated_sharding=replicated_sharding,
                    distributed_sharding=distributed_sharding
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

        # Batch sample X|Y for all Y
        n_batch_sample = 100
        n_batches_sample = int(config.data.n_data / n_batch_sample)
        X_Y = []
        with trange(
            n_batches_sample, desc="Sampling X|Y", colour="magenta"
        ) as steps:
            for _, _Y, _A in zip(
                steps, 
                jnp.split(Y, n_batches_sample),
                jnp.split(A, n_batches_sample)
            ):
                # Generate latents from q(x|y)
                sampler = get_x_y_sampler(
                    ema_flow if config.train.use_ema else flow, 
                    cov_y=cov_y, 
                    mu_x=mu_x, 
                    cov_x=cov_x, 
                    mode=config.train.mode,
                    max_steps=config.train.max_steps,
                    sde_type=config.train.sde_type,
                    sampling_mode=config.train.sampling_mode, 
                    postprocess_fn=postprocess_fn
                )

                keys = jr.split(key_sample, n_batch_sample) # NOTE: split key
                X_Y_ = jax.vmap(sampler)(keys, _Y, _A)

                X_Y.append(X_Y_)

        X_Y = jnp.concatenate(X_Y)

        if config.train.clip_x_y:
            X_Y = clip_latents(X_Y, config.train.x_clip_limit) 

        # Generate latents from q(x)
        sampler = get_x_sampler(
            ema_flow if config.train.use_ema else flow, 
            sampling_mode=config.train.sampling_mode, 
            sde_type=config.train.sde_type,
            postprocess_fn=postprocess_fn
        )
        X_ = jax.vmap(sampler)(keys)

        # Plot latents
        plot_samples(X, X_Y, Y, X_, iteration=k + 1, save_dir=save_dir)

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