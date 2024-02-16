from typing import Any, Iterator

from datasets import Dataset


class DataLoader:
    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        infinite: bool = False,
        shuffle: bool = False,
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.infinite = infinite
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Any]:
        while True:
            if self.shuffle:
                self.ds = self.ds.shuffle(load_from_cache_file=False)
            yield from self.ds.iter(batch_size=self.batch_size)
            if not self.infinite:
                break
