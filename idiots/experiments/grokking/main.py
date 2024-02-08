import torch
import torch.func as ft
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger

from idiots.dataset.algorithmic import binary_op_loaders
from idiots.experiments.grokking.model import TransformerSingleOutput
from idiots.utils import metrics, take_with_repeat, tree_flatten

task: str = "x + y (mod 47)"
log_every: int = 100
eval_every: int = 1000
warmup_steps: int = 10
train_batch_size: int = 128
test_batch_size: int = 16
train_percentage: float = 0.3
weight_decay: float = 0.1
steps: int = int(1e5)


def lr_schedule(step: int) -> float:
    match step:
        case step if step < warmup_steps:
            return step / warmup_steps
        case _:
            return 1


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    for x, y in test_loader:
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        acc = torch.argmax(y_pred, dim=-1) == y
        metrics.log(eval_loss=loss, eval_accuracy=acc)
    model.train()


def dots(f, theta, test_loader, n_batches: int) -> int:
    xs = [x for _, (x, _) in take_with_repeat(test_loader, n_batches)]
    xs = torch.cat(xs, dim=0)

    df_dtheta = ft.jacrev(f)
    jac = df_dtheta(theta, xs)
    jac_flat, _ = tree_flatten(jac)
    jac_flat = [j.flatten(end_dim=1).flatten(start_dim=1) for j in jac_flat]
    jac_flat = torch.cat(jac_flat, dim=1).float()

    return torch.linalg.matrix_rank(jac_flat).item()


def main(_):
    logger = TensorBoardLogger(root_dir="logs", name="grokking")
    fabric = Fabric(precision="32-true", loggers=logger)
    fabric.launch()

    dataset, train_loader, test_loader = binary_op_loaders(
        task, train_batch_size, test_batch_size, train_percentage
    )
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    model = TransformerSingleOutput(
        num_tokens=dataset.vocab_size,
        max_seq_len=dataset.seq_len,
        dim=64,
        depth=2,
        heads=2,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=weight_decay
    )
    model, optimizer = fabric.setup(model, optimizer)
    theta = dict(model.named_parameters())

    def model_f(theta, x):
        return ft.functional_call(model, theta, x)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    for step, (epoch, (x, y)) in enumerate(
        take_with_repeat(train_loader, steps), start=0
    ):
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        fabric.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        acc = torch.argmax(y_pred, dim=-1) == y
        metrics.log(loss=loss, accuracy=acc)

        if step % log_every == 0:
            [losses, accuracies] = metrics.collect("loss", "accuracy")
            fabric.log_dict(
                {
                    "train/loss": torch.stack(losses).mean(),
                    "train/accuracy": torch.cat(accuracies).float().mean(),
                    "epoch": epoch,
                },
                step,
            )

        if step % eval_every == 0:
            evaluate(model, test_loader)
            [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
            fabric.log_dict(
                {
                    "eval/loss": torch.stack(losses).mean(),
                    "eval/accuracy": torch.cat(accuracies).float().mean(),
                    "eval/dots": dots(model_f, theta, test_loader, 1),
                    "epoch": epoch,
                },
                step,
            )


if __name__ == "__main__":
    main(None)
