import jax
import jax.numpy as jnp
import optax
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
train_percentage: float = 0.3
weight_decay: float = 0.1
steps: int = int(1e5)


class TrainState(train_state.TrainState):
    ...


def init():
    rng = jax.random.PRNGKey(0)
    writer = SummaryWriter(log_dir=str(next_dir("logs/grokking")))

    ds_train, ds_test = binary_op_splits(task, train_percentage)
    train_loader = DataLoader(ds_train, train_batch_size, infinite=True)
    test_loader = DataLoader(ds_test, test_batch_size, shuffle=False)

    model = TransformerSingleOutput(
        d_model=128,
        n_layers=2,
        n_heads=2,
        vocab_size=ds_train.features["y"].num_classes,
        max_len=ds_train.features["x"].length,
    )
    params = model.init(rng, next(iter(train_loader))[1]["x"])
    tx = optax.adamw(1e-3, b1=0.9, b2=0.98, weight_decay=weight_decay)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    print(f"Number of parameters: {num_params(state.params):,}")
    return state, train_loader, test_loader, writer


def main(_):
    state, train_loader, test_loader, writer = init()

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

    train_iter = iter(train_loader)

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
            for batch in test_loader:
                logs = eval_step(state, batch)
                metrics.log(**logs)
            [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
            log_dict(
                {
                    "eval/loss": jnp.concatenate(losses).mean().item(),
                    "eval/accuracy": jnp.concatenate(accuracies).mean().item(),
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


# def dots(f, theta, test_loader, n_batches: int) -> int:
#     xs = [x for _, (x, _) in take_with_repeat(test_loader, n_batches)]
#     xs = torch.cat(xs, dim=0)

#     df_dtheta = ft.jacrev(f)
#     jac = df_dtheta(theta, xs)
#     jac_flat, _ = tree_flatten(jac)
#     jac_flat = [j.flatten(end_dim=1).flatten(start_dim=1) for j in jac_flat]
#     jac_flat = torch.cat(jac_flat, dim=1).float()

#     return torch.linalg.matrix_rank(jac_flat).item()


if __name__ == "__main__":
    main(None)
