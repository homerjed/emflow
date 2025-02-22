import math
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


def get_mnist_config():
    config = ConfigDict()

    # Train
    train = config.train = ConfigDict()
    # > Train on latents first to check flow works (discarding it)
    train.test_on_latents      = False
    # > Iterations
    train.em_iterations        = 256 #64
    train.diffusion_iterations = 10_000
    train.n_batch              = 50 * jax.local_device_count()
    train.loss_type            = "mse"
    train.time_schedule        = identity
    # > Optimiser
    train.lr                   = 1e-5
    train.optimiser            = soap
    train.use_lr_schedule      = False
    train.initial_lr           = 1e-3
    train.n_epochs_warmup      = 1
    train.ppca_pretrain        = True
    train.n_pca_iterations     = 16
    # > Sampling
    train.clip_x_y             = False # Clip sampled latents
    train.x_clip_limit         = 4.
    train.re_init_opt_state    = True
    train.sampling_mode        = "ode"       # ODE, SDE or DDIM
    train.sde_type             = "zero-ends" # SDE of flow ODE
    train.mode                 = "cg"      # CG mode or not
    train.max_steps            = 2
    # > EMA and minibatching
    train.use_ema              = True
    train.ema_rate             = 0.999
    train.accumulate_gradients = True
    train.n_minibatches        = 4

    # Data
    data = config.data = ConfigDict()
    data.dataset              = "mnist"
    data.img_size             = 28
    data.img_shape            = (1, 28, 28)
    data.data_dim             = math.prod(data.img_shape) # Same dimension, just masked partly
    data.latent_dim           = math.prod(data.img_shape)
    data.mask_fraction        = 0.5
    data.n_data               = 1000
    data.sigma_y              = 0.005

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


    assert config.data.dataset in ["gmm", "moons", "blob", "double-blob", "mnist"]
    assert config.train.sampling_mode in ["ddim", "ode", "sde"]
    assert config.train.sde_type in ["non-singular", "zero-ends", "singular", "gamma"]
    assert config.train.mode in ["full", "cg"]

    return config