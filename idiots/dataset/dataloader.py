from typing import Any, Iterator

import jax
import jax.numpy as jnp
from datasets import Dataset


class DataLoader:
    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        infinite: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.key = jax.random.PRNGKey(seed)
        self.perm = jnp.arange(len(ds))
        self.batch_size = batch_size
        self.infinite = infinite
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Buffer everything in-memory
        self.buffer = {}
        for colunm in ds.column_names:
            self.buffer[colunm] = jax.device_put(ds[colunm])

    def __iter__(self) -> Iterator[Any]:
        while True:
            if self.shuffle:
                self.key, subkey = jax.random.split(self.key)
                self.perm = jax.random.permutation(subkey, self.perm)
                for k, v in self.buffer.items():
                    self.buffer[k] = v[self.perm]

            for i in range(0, len(self.perm), self.batch_size):
                if i + self.batch_size > len(self.perm) and self.drop_last:
                    break
                yield {k: v[i : i + self.batch_size] for k, v in self.buffer.items()}

            if not self.infinite:
                break
