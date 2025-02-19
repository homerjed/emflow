import jax
import jax.numpy as jnp
import diffrax as dfx
from ml_collections import ConfigDict

from rf import identity
from soap import soap


def get_blob_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    train.test_on_latents      = False
    train.em_iterations        = 64
    train.diffusion_iterations = 5_000
    train.n_batch              = 5_000
    train.loss_type            = "mse"
    train.time_schedule        = identity
    train.lr                   = 1e-5
    train.optimiser            = soap
    train.use_lr_schedule      = False
    train.initial_lr           = 1e-3
    train.n_epochs_warmup      = 1
    train.ppca_pretrain        = True
    train.n_pca_iterations     = 512
    train.clip_x_y             = False # Clip sampled latents
    train.x_clip_limit         = 4.
    train.re_init_opt_state    = True
    train.sampling_mode        = "ode"        # ODE, SDE or DDIM
    train.sde_type             = "zero-ends"  # SDE of flow ODE
    train.mode                 = "full"       # CG mode or not
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = False
    train.n_minibatches        = 4

    # Model
    model = config.model = ConfigDict()
    model.width_size           = 256 
    model.depth                = 3 
    model.activation           = jax.nn.gelu 
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea
    model.time_embedding_dim   = 64

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "blob"
    data.data_dim             = 2
    data.n_data               = 100_000
    data.sigma_y              = 0.4 # Tiny eigenvalues may have been numerically unstable?
    data.cov_y                = jnp.eye(data_dim) * jnp.square(sigma_y)

    return config


def get_gmm_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    # > Test on latents first
    train.test_on_latents      = False
    # > Iterations
    train.em_iterations        = 64
    train.diffusion_iterations = 8_000
    train.n_batch              = 5_000
    train.loss_type            = "mse"
    train.time_schedule        = identity
    # > Optimiser
    train.lr                   = 1e-5
    train.optimiser            = soap
    train.use_lr_schedule      = False
    train.initial_lr           = 1e-3
    train.n_epochs_warmup      = 1
    train.ppca_pretrain        = True
    train.n_pca_iterations     = 512
    # > Sampling
    train.clip_x_y             = False # Clip sampled latents
    train.x_clip_limit         = 4.
    train.re_init_opt_state    = True
    train.sampling_mode        = "ode"        # ODE, SDE or DDIM
    train.sde_type             = "zero-ends"  # SDE of flow ODE
    train.mode                 = "full"       # CG mode or not
    # > EMA and minibatching
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = False
    train.n_minibatches        = 4

    # Model
    model = config.model = ConfigDict()
    model.width_size           = 256 
    model.depth                = 3 
    model.activation           = jax.nn.gelu 
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea
    model.time_embedding_dim   = 64

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "gmm"
    data.data_dim             = 2
    data.n_data               = 100_000
    # > Corruption noise
    data.sigma_y              = 0.2 # Tiny eigenvalues may have been numerically unstable?
    data.cov_y                = jnp.eye(data_dim) * jnp.square(sigma_y)

    return config