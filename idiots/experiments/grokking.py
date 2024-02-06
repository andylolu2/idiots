import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from x_transformers import Decoder, TransformerWrapper

from idiots.dataset.algorithmic import binary_op_loaders
from idiots.utils import metrics, take_with_repeat

task: str = "x / y (mod 97)"
log_every: int = 100
eval_every: int = 100
warmup_steps: int = 10
batch_size: int = 512
train_percentage: float = 0.5
weight_decay: float = 0
steps: int = int(1e5)


def lr_schedule(step: int) -> float:
    match step:
        case step if step < warmup_steps:
            return step / warmup_steps
        case _:
            return 1


def accuracy(
    logits: torch.Tensor, y: torch.Tensor, ignore_idx: int = -100
) -> torch.Tensor:
    return (logits.argmax(dim=-1) == y)[y != ignore_idx].float().mean()


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    for x, y in test_loader:
        logits = model(x)
        loss = F.cross_entropy(torch.flatten(logits, end_dim=-2), torch.flatten(y))
        acc = accuracy(logits, y)
        metrics.log(eval_loss=loss, eval_accuracy=acc)
    model.train()


def main(_):
    logger = TensorBoardLogger(root_dir="logs", name="grokking")
    fabric = Fabric(precision="32-true", loggers=logger)
    fabric.launch()

    dataset, train_loader, test_loader = binary_op_loaders(
        task, batch_size, train_percentage
    )
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    model = TransformerWrapper(
        num_tokens=dataset.vocab_size,
        max_seq_len=dataset.seq_len,
        attn_layers=Decoder(dim=128, depth=2, heads=4),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=weight_decay
    )
    model, optimizer = fabric.setup(model, optimizer)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    for step, (epoch, (x, y)) in enumerate(
        take_with_repeat(train_loader, steps), start=1
    ):
        logits = model(x)
        loss = F.cross_entropy(torch.flatten(logits, end_dim=-2), torch.flatten(y))
        fabric.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        acc = accuracy(logits, y)
        metrics.log(loss=loss, accuracy=acc)

        if step % log_every == 0:
            [losses, accuracies] = metrics.collect("loss", "accuracy")
            fabric.log_dict(
                {
                    "train/loss": torch.stack(losses).mean(),
                    "train/accuracy": torch.stack(accuracies).mean(),
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
                    "eval/accuracy": torch.stack(accuracies).mean(),
                    "epoch": epoch,
                },
                step,
            )


if __name__ == "__main__":
    main(None)
