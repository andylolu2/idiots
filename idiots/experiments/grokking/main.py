import jax
import jax.numpy as jnp
import neural_tangents as nt
import optax
from einops import rearrange
from flax.training import train_state
from tensorboardX import SummaryWriter

from idiots.dataset.algorithmic import DataLoader, binary_op_splits
from idiots.experiments.grokking.model import TransformerSingleOutput
from idiots.utils import log_dict, metrics, next_dir, num_params

task: str = "x + y (mod 47)"
log_every: int = 100
eval_every: int = 1000
warmup_steps: int = 10
train_batch_size: int = 256
test_batch_size: int = 256
dots_sample_size: int = 64
train_percentage: float = 0.4
weight_decay: float = 0.1
steps: int = int(1e5)


class TrainState(train_state.TrainState):
    ...


@jax.jit
def dots(state: TrainState, x):
    kernel_fn = nt.empirical_kernel_fn(
        state.apply_fn,
        trace_axes=(),
        vmap_axes=0,
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,
    )
    k = kernel_fn(x, None, "ntk", state.params)
    k = rearrange(k, "b1 b2 d1 d2 -> (b1 d1) (b2 d2)")
    return jnp.linalg.matrix_rank(k)


def init():
    rng = jax.random.PRNGKey(0)
    writer = SummaryWriter(log_dir=str(next_dir("logs/grokking")))

    ds_train, ds_test = binary_op_splits(task, train_percentage)

    model = TransformerSingleOutput(
        d_model=128,
        n_layers=2,
        n_heads=4,
        vocab_size=ds_train.features["y"].num_classes,
        max_len=ds_train.features["x"].length,
    )
    params = model.init(rng, ds_train["x"][:1])
    tx = optax.adamw(1e-3, b1=0.9, b2=0.98, weight_decay=weight_decay)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    print(f"Number of parameters: {num_params(state.params):,}")
    return state, ds_train, ds_test, writer


def main(_):
    state, ds_train, ds_test, writer = init()

    @jax.jit
    def train_step(state: TrainState, batch) -> tuple[TrainState, dict]:
        def forward(params, x, y):
            y_pred = state.apply_fn(params, x)
            losses = optax.softmax_cross_entropy_with_integer_labels(y_pred, y)
            loss = jnp.mean(losses)
            return loss, (losses, y_pred)

        grads, (losses, y_pred) = jax.grad(forward, has_aux=True)(
            state.params, batch["x"], batch["y"]
        )
        acc = jnp.argmax(y_pred, axis=-1) == batch["y"]
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        logs = {"loss": losses, "accuracy": acc}
        return new_state, logs

    @jax.jit
    def eval_step(state: TrainState, batch) -> dict:
        y_pred = state.apply_fn(state.params, batch["x"])
        loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, batch["y"])
        acc = jnp.argmax(y_pred, axis=-1) == batch["y"]
        return {"eval_loss": loss, "eval_accuracy": acc}

    train_iter = iter(
        DataLoader(ds_train, train_batch_size, shuffle=True, infinite=True)
    )

    while state.step < steps:
        state, logs = train_step(state, next(train_iter))
        metrics.log(**logs)

        if state.step % log_every == 0:
            [losses, accuracies] = metrics.collect("loss", "accuracy")
            log_dict(
                {
                    "train/loss": jnp.concatenate(losses).mean().item(),
                    "train/accuracy": jnp.concatenate(accuracies).mean().item(),
                },
                state.step,
                writer,
            )

        if state.step % eval_every == 0:
            for batch in DataLoader(ds_test, test_batch_size):
                logs = eval_step(state, batch)
                metrics.log(**logs)
            [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
            log_dict(
                {
                    "eval/loss": jnp.concatenate(losses).mean().item(),
                    "eval/accuracy": jnp.concatenate(accuracies).mean().item(),
                    "eval/dots": dots(state, ds_test["x"][:dots_sample_size]).item(),
                },
                state.step,
                writer,
            )


# def lr_schedule(step: int) -> float:
#     match step:
#         case step if step < warmup_steps:
#             return step / warmup_steps
#         case _:
#             return 1


if __name__ == "__main__":
    main(None)
