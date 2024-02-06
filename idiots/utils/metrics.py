from collections import defaultdict, deque

import torch


class _Metrics:
    def __init__(self, max_len: int | None = None):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self.history[k].append(v)

    def clear(self):
        self.history.clear()

    def collect(self, *keys: str, clear: bool = True) -> tuple[list, ...]:
        if clear:
            res = tuple(list(self.history.pop(k)) for k in keys)
        else:
            res = tuple(list(self.history[k]) for k in keys)
        return res

    def asdict(self) -> dict[str, list]:
        return {k: list(v) for k, v in self.history.items()}


metrics = _Metrics()
