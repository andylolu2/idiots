from torch import nn


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
