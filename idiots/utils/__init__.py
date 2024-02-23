import re
from pathlib import Path

import jax

from .metrics import metrics
from .optimizers import get_optimizer


def num_params(params):
    return sum(x.size for x in jax.tree_leaves(params))


def next_dir(path: str | Path, prefix: str = "exp") -> Path:
    path = Path(path)
    if not path.exists():
        return path / f"{prefix}1"

    largest = 0
    for p in path.iterdir():
        if not p.is_dir():
            continue
        match = re.match(f"{prefix}(\d+)", p.name)
        if match:
            largest = max(largest, int(match.group(1)))
    return path / f"{prefix}{largest + 1}"
