from itertools import product

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
}


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


def binary_op_splits(op: str = "x + y (mod 97)", train_percentage: float = 0.5):
    ds = binary_op_dataset(op).with_format("jax")
    ds_split = ds.train_test_split(train_size=train_percentage, shuffle=True)
    return ds_split["train"], ds_split["test"]


if __name__ == "__main__":
    from idiots.dataset.dataloader import DataLoader

    ds_train, ds_test = binary_op_splits("x + y (mod 47)", 0.3)
    print(ds_train.features)
    for item in DataLoader(ds_test, 32):
        print(item["x"].shape, item["y"].shape)
        print(item["x"].dtype, item["y"].dtype)
        break
