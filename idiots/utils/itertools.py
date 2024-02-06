from typing import Iterable, TypeVar

T = TypeVar("T")


def take_with_repeat(iter: Iterable[T], n: int) -> Iterable[tuple[int, T]]:
    i = 0
    epoch = 0
    while True:
        for item in iter:
            if i >= n:
                return
            yield epoch, item
            i += 1
        epoch += 1
