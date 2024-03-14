import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
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
from idiots.utils import num_params

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", short_name="c", lock_config=True)
flags.mark_flags_as_required(["config"])


def main(_):
    config = FLAGS.config
    apply_fn, params, ds_train, ds_test, mngr = init(config)

    logging.info("Number of parameters: %d", num_params(params))

    xs_train, ys_train = ds_train["x"], ds_train["y"]
    ys_train = jax.nn.one_hot(ys_train, ds_train.features["y"].num_classes)
    xs_train, ys_train = jax.device_put(xs_train), jax.device_put(ys_train)

    def update_fn(params):
        def loss_fn(params):
            ys_pred = apply_fn(params, xs_train)
            return optax.l2_loss(ys_pred, ys_train).mean()

        grad = jax.grad(loss_fn)(params)
        update = jax.tree_map(
            lambda g, p: -(g + config.weight_decay * p),
            grad,
            params,
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
