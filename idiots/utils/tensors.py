import torch


def n_unsqueeze(x: torch.Tensor, dim: int = 0, n: int = 1) -> torch.Tensor:
    for _ in range(n):
        x = x.unsqueeze(dim)
    return x
