import jax
import jax.numpy as jnp




def log_prob_exact(t, y, args):
    """ 
        Compute trace directly. 
    """
    y, _ = y
    func, data_shape = args

    fn = lambda y: func(y.reshape(data_shape), t)
    f, f_vjp = jax.vjp(fn, y)  

    (dfdy,) = jax.vmap(f_vjp)(jnp.eye(y.size)) 
    log_prob = jnp.trace(dfdy)

    return f, log_prob


def prior_log_prob(z):
    return jnp.sum(jax.scipy.stats.norm.logpdf(z))


@eqx.filter_jit
def log_likelihood(
    sgm: SGM, x: XArray
) -> tuple[XArray, Scalar]:
    sgm = eqx.nn.inference_mode(sgm, True)

    def ode(y, t):
        return sgm.sde.reverse_ode(sgm.net(t, y), y, t)

    # Likelihood from solving initial value problem
    sol = dfx.diffeqsolve(
        dfx.ODETerm(log_prob_exact),
        dfx.Heun(), 
        t0=0.,
        t1=t1, 
        dt0=0.01, 
        y0=(x.flatten(), 0.), # Data and initial change in log_prob
        args=(ode, sgm.x_shape),
    ) 
    (z,), (delta_log_likelihood,) = sol.ys
    p_z = prior_log_prob(z)
    log_p_x = p_z + delta_log_likelihood 
    return z, log_p_x