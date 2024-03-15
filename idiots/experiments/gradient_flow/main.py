import jax
import jax.numpy as jnp
import optax
from absl import app, flags, logging
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    TqdmProgressMeter,
    Tsit5,
    diffeqsolve,
)
from ml_collections import config_flags

from idiots.experiments.gradient_flow.init import init
from idiots.experiments.grokking.training import loss_fn
from idiots.utils import num_params

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", short_name="c", lock_config=True)
flags.mark_flags_as_required(["config"])


def main(_):
    config = FLAGS.config
    apply_fn, params, ds_train, ds_test, mngr = init(config)

    logging.info("Number of parameters: %d", num_params(params))

    xs_train, ys_train = ds_train["x"], ds_train["y"]
    xs_train, ys_train = jax.device_put(xs_train), jax.device_put(ys_train)

    def update_fn(params):
        def forward(params):
            ys_pred = apply_fn(params, xs_train)
            return loss_fn(ys_pred, ys_train, variant=config.loss_variant).mean()

        grad = jax.grad(forward)(params)
        update = jax.tree_map(
            lambda g, p: -(g + config.weight_decay * p),
            grad,
            params,
        )

        if config.fixed_weight_norm:
            # u_fixed = u - (u . p^hat) p^hat
            p_norm = optax.global_norm(params)
            p_hat = jax.tree_map(lambda p: p / p_norm, params)
            u_dot_p_hat = sum(
                jax.tree_util.tree_leaves(jax.tree_map(jnp.vdot, update, p_hat))
            )
            update = jax.tree_map(
                lambda u, p_hat: u - u_dot_p_hat * p_hat, update, p_hat
            )

        return update

    term = ODETerm(lambda t, ps, args: update_fn(ps))
    solver = Tsit5()
    save_at = SaveAt(ts=jnp.arange(0, config.T + config.save_every, config.save_every))
    step_size_controller = PIDController(
        rtol=config.ode.rtol,
        atol=config.ode.atol,
        pcoeff=config.ode.pcoeff,
        icoeff=config.ode.icoeff,
    )

    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=config.T,
        dt0=config.ode.dt0,
        y0=params,
        saveat=save_at,
        stepsize_controller=step_size_controller,
        max_steps=None,
        progress_meter=TqdmProgressMeter(),
    )

    for i, t in enumerate(sol.ts):
        param = jax.tree_map(lambda x: x[i], sol.ys)
        mngr.save(int(t), param)

    mngr.wait_until_finished()


if __name__ == "__main__":
    app.run(main)
