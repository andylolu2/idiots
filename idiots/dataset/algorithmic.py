from itertools import product
from math import factorial

from datasets import ClassLabel, Dataset, Features, Sequence

OPERATIONS = {
    "x + y (mod 97)": {
        "fn": lambda x, y: (x + y) % 97,
        "n_classes": 97,
    },
    "x / y (mod 97)": {
        "fn": lambda x, y: (x * pow(y, 95, 97)) % 97,
        "n_classes": 97,
    },
    "x / y (mod 47)": {
        "fn": lambda x, y: (x * pow(y, 45, 47)) % 47,
        "n_classes": 47,
    },
    "x + y (mod 47)": {
        "fn": lambda x, y: (x + y) % 47,
        "n_classes": 47,
    },
    "xy (S5)": {
        "fn": lambda x, y: permuatation_op(x, y, 5),
        "n_classes": 120,
    },
}


def permuatation_op(x: int, y: int, n: int) -> int:
    """S_n group operation."""

    def _int_to_perm(x: int, n: int) -> list[int]:
        """int to member of S_n."""
        if n == 1:
            assert x == 0
            return [0]

        last_digit = x // factorial(n - 1)
        others = _int_to_perm(x % factorial(n - 1), n - 1)
        for i in range(n - 1):
            if others[i] >= last_digit:
                others[i] += 1
        return others + [last_digit]

    def _perm_to_int(p: list[int], n: int) -> int:
        """member of S_n to int."""
        if n == 1:
            assert p == [0]
            return 0
        for i in range(n - 1):
            if p[i] > p[-1]:
                p[i] -= 1
        return p[-1] * factorial(n - 1) + _perm_to_int(p[:-1], n - 1)

    x_perm = _int_to_perm(x, n)
    y_perm = _int_to_perm(y, n)
    z_perm = [y_perm[x_perm[i]] for i in range(n)]
    z = _perm_to_int(z_perm, n)
    return z


def binary_op_dataset(op: str):
    """Binary operation dataset.

    Features:
        x: (4,) int32, of the form "a ? b ="
        y: () int32, result of the operation
    """
    fn = OPERATIONS[op]["fn"]
    n_classes = OPERATIONS[op]["n_classes"]
    OP, EQ = n_classes, n_classes + 1

    x, y = [], []
    for a, b in product(range(n_classes), repeat=2):
        x.append([a, OP, b, EQ])
        y.append(fn(a, b))
    class_label = ClassLabel(
        num_classes=n_classes + 2,
        names=[str(i) for i in range(n_classes)] + ["?", "="],
    )
    return Dataset.from_dict(
        {"x": x, "y": y},
        Features({"x": Sequence(class_label, length=4), "y": class_label}),
    )


def binary_op_splits(
    op: str = "x + y (mod 97)", train_percentage: float = 0.5, seed: int = 0
):
    ds = binary_op_dataset(op).with_format("numpy")
    ds_split = ds.train_test_split(train_size=train_percentage, shuffle=True, seed=seed)
    return ds_split["train"], ds_split["test"]


if __name__ == "__main__":
    from time import perf_counter

    from idiots.dataset.dataloader import DataLoader

    ds_train, ds_test = binary_op_splits("xy (S5)", 0.3)
    print(ds_train.features)
    for item in DataLoader(ds_test, 32):
        print(item["x"].shape, item["y"].shape)
        print(item["x"].dtype, item["y"].dtype)
        break

    # Performance test
    start = perf_counter()
    n_iters = 0
    for batch in DataLoader(ds_train, 32):
        n_iters += 1
    print(f"Time: {perf_counter() - start:.4f} seconds")
    print(f"Time/step: {(perf_counter() - start) / 100:.4f} seconds")
