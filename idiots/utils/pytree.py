from typing import Any, Callable

from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten


def tree_map(func: Callable[..., Any], *trees: PyTree) -> PyTree:
    flat_args, spec = zip(*(tree_flatten(tree) for tree in trees))
    return tree_unflatten([func(*args) for args in zip(*flat_args)], spec[0])
