import yaml
import jax
import diffrax as dfx
import optax
from ml_collections import ConfigDict

from rf import identity
from soap import soap


def save_config(config: ConfigDict, filename: str) -> None:
    # Save a config to a yaml file
    with open(filename, 'w') as f:
        yaml.dump(config.to_dict(), f)


def load_config(filename: str) -> ConfigDict:
    # Load a config to a yaml file
    with open(filename, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigDict(config_dict)


def get_blob_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    # > Train on latents first to check flow works (discarding it)
    train.test_on_latents      = False
    # > Iterations
    train.em_iterations        = 256 #64
    train.diffusion_iterations = 5_000
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
    train.n_pca_iterations     = 10 
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

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "blob"
    data.data_dim             = 2
    data.latent_dim           = 2 # No corruption
    data.n_data               = 100_000
    data.sigma_y              = 0.1 #0.05

    # Model
    model = config.model = ConfigDict()
    model.model_type           = "resnet"
    model.data_dim             = data.latent_dim
    model.width_size           = 256 
    model.depth                = 3 
    model.activation           = jax.nn.gelu 
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea
    model.time_embedding_dim   = 64

    assert config.data.dataset in ["gmm", "moons", "blob", "double-blob"]
    assert config.train.sampling_mode in ["ddim", "ode", "sde"]
    assert config.train.sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert config.train.mode in ["full", "cg"]

    return config


def get_double_blob_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    # > Train on latents first to check flow works (discarding it)
    train.test_on_latents      = False
    # > Iterations
    train.em_iterations        = 256 #64
    train.diffusion_iterations = 5_000
    train.n_batch              = 5_000
    train.loss_type            = "mse"
    train.time_schedule        = identity
    # > Optimiser
    train.lr                   = 1e-6
    train.optimiser            = optax.adam #soap
    train.use_lr_schedule      = True
    train.initial_lr           = 1e-3
    train.n_epochs_warmup      = 1
    train.ppca_pretrain        = True
    train.n_pca_iterations     = 512
    # > Sampling
    train.clip_x_y             = False # Clip sampled latents
    train.x_clip_limit         = 4.
    train.re_init_opt_state    = True
    train.sampling_mode        = "ode"       # ODE, SDE or DDIM
    train.sde_type             = "zero-ends" # SDE of flow ODE
    train.mode                 = "full"      # CG mode or not
    # > EMA and minibatching
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = False
    train.n_minibatches        = 4

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "double-blob"
    data.data_dim             = 2
    data.n_data               = 100_000
    data.sigma_y              = 0.3 #0.05

    # Model
    model = config.model = ConfigDict()
    model.model_type           = "resnet"
    model.data_dim             = data.data_dim
    model.width_size           = 256 
    model.depth                = 3 
    model.activation           = jax.nn.gelu 
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea
    model.time_embedding_dim   = 64

    assert config.data.dataset in ["gmm", "moons", "blob", "double-blob"]
    assert config.train.sampling_mode in ["ddim", "ode", "sde"]
    assert config.train.sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert config.train.mode in ["full", "cg"]

    return config


def get_moons_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    # > Train on latents first to check flow works (discarding it)
    train.test_on_latents      = False
    # > Iterations
    train.em_iterations        = 256 #64
    train.diffusion_iterations = 5_000
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
    train.n_pca_iterations     = 256 
    # > Sampling
    train.clip_x_y             = False # Clip sampled latents
    train.x_clip_limit         = 4.
    train.re_init_opt_state    = True
    train.sampling_mode        = "ode"        # ODE, SDE or DDIM
    train.sde_type             = "zero-ends"  # SDE of flow ODE
    train.mode                 = "cg"       # CG mode or not
    # > EMA and minibatching
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = False
    train.n_minibatches        = 4

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "moons"
    data.data_dim             = 2
    data.latent_dim           = 2 # No corruption
    data.n_data               = 100_000
    data.sigma_y              = 0.1 #0.05

    # Model
    model = config.model = ConfigDict()
    model.model_type           = "resnet"
    model.data_dim             = data.latent_dim
    model.width_size           = 256 
    model.depth                = 3 
    model.activation           = jax.nn.gelu 
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea
    model.time_embedding_dim   = 64

    assert config.data.dataset in ["gmm", "moons", "blob", "double-blob"]
    assert config.train.sampling_mode in ["ddim", "ode", "sde"]
    assert config.train.sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert config.train.mode in ["full", "cg"]

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
    train.n_pca_iterations     = 32
    # > Sampling
    train.clip_x_y             = False # Clip sampled latents
    train.x_clip_limit         = 4.
    train.re_init_opt_state    = True
    train.sampling_mode        = "ode"       # ODE, SDE or DDIM
    train.sde_type             = "zero-ends" # SDE of flow ODE
    train.mode                 = "full"      # CG mode or not
    # > EMA and minibatching
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = True
    train.n_minibatches        = 4

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "gmm"
    data.data_dim             = 2
    data.latent_dim           = 2 
    data.n_data               = 100_000
    data.sigma_y              = 0.2

    # Model
    model = config.model = ConfigDict()
    model.data_dim             = data.latent_dim
    model.width_size           = 256 
    model.depth                = 3 
    model.activation           = jax.nn.gelu 
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) 
    model.time_embedding_dim   = 64
    model.model_type           = "resnet"

    assert config.data.dataset in ["gmm", "moons", "blob", "double-blob"]
    assert config.train.sampling_mode in ["ddim", "ode", "sde"]
    assert config.train.sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert config.train.mode in ["full", "cg"]

    return config


def get_mnist_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    # > Train on latents first to check flow works (discarding it)
    train.test_on_latents      = True
    # > Iterations
    train.em_iterations        = 256 #64
    train.diffusion_iterations = 5_000
    train.n_batch              = 32
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
    train.sampling_mode        = "ode"       # ODE, SDE or DDIM
    train.sde_type             = "zero-ends" # SDE of flow ODE
    train.mode                 = "full"      # CG mode or not
    # > EMA and minibatching
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = True
    train.n_minibatches        = 4

    # Model
    model = config.model = ConfigDict()
    model.model_type           = "dit"
    model.img_size             = 28
    model.channels             = 1
    model.patch_size           = 4 
    model.depth                = 4
    model.n_heads              = 4
    model.embed_dim            = 128
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) # For ODE NOTE: this eps supposed to be bad idea

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "mnist"
    data.data_dim             = model.img_size ** 2
    data.n_data               = 100 # 60_000 NOTE: match this elsewhere
    data.sigma_y              = 0.05

    assert config.data.dataset in ["gmm", "moons", "blob", "double-blob", "mnist"]
    assert config.train.sampling_mode in ["ddim", "ode", "sde"]
    assert config.train.sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert config.train.mode in ["full", "cg"]

    return config


def get_spiral_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    # > Train on latents first to check flow works (discarding it)
    train.test_on_latents      = False
    # > Iterations
    train.em_iterations        = 256 #64
    train.diffusion_iterations = 5_000
    train.n_batch              = 32
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
    train.sampling_mode        = "ode"       # ODE, SDE or DDIM
    train.sde_type             = "zero-ends" # SDE of flow ODE
    train.mode                 = "cg"        # CG mode or not
    # > EMA and minibatching
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = True
    train.n_minibatches        = 4

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "spiral"
    data.data_dim             = 2
    data.latent_dim           = 3
    data.n_data               = 100_000
    data.sigma_y              = 0.01

    # Model
    model = config.model = ConfigDict()
    model.data_dim             = data.latent_dim
    model.width_size           = 256 
    model.depth                = 3 
    model.activation           = jax.nn.gelu 
    model.soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) 
    model.time_embedding_dim   = 64
    model.model_type           = "resnet"

    assert config.data.dataset in ["gmm", "spiral", "moons", "blob", "double-blob", "mnist"]
    assert config.train.sampling_mode in ["ddim", "ode", "sde"]
    assert config.train.sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert config.train.mode in ["full", "cg"]

    return config 


    # # Train
    # test_on_latents      = False
    # em_iterations        = 64
    # diffusion_iterations = 5_000
    # n_batch              = 5_000
    # loss_type            = "mse"
    # time_schedule        = identity
    # lr                   = 1e-5
    # optimiser            = soap
    # use_lr_schedule      = False
    # initial_lr           = 1e-3
    # n_epochs_warmup      = 1
    # ppca_pretrain        = True
    # n_pca_iterations     = 512
    # clip_x_y             = False # Clip sampled latents
    # x_clip_limit         = 4.
    # re_init_opt_state    = True
    # sampling_mode        = "ode"        # ODE, SDE or DDIM
    # sde_type             = "zero-ends"  # SDE of flow ODE
    # mode                 = "full"       # CG mode or not
    # use_ema              = True
    # ema_rate             = 0.999
    # accumulate_gradients = True
    # n_minibatches        = 4

    # # Model
    # width_size           = 256 
    # depth                = 3 
    # activation           = jax.nn.gelu 
    # soln_kwargs          = dict(t0=0., dt0=0.01, t1=1., solver=dfx.Euler()) 
    # time_embedding_dim   = 64

    # # Data
    # dataset              = "moons"
    # data_dim             = 2
    # n_data               = 100_000
    # sigma_y              = 0.05
    # cov_y                = jnp.eye(data_dim) * jnp.square(sigma_y)