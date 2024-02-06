from itertools import product

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

OPERATIONS = {
    "x + y (mod 97)": {
        "fn": lambda x, y: (x + y) % 97,
        "n_classes": 97,
    },
    "x / y (mod 97)": {
        "fn": lambda x, y: (x * pow(y, 95, 97)) % 97,
        "n_classes": 97,
    },
}


def binary_op_loaders(op: str, batch_size: int, train_percentage: float):
    dataset = BinaryOp(op)
    train_size = int(len(dataset) * train_percentage)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return dataset, train_loader, test_loader


class BinaryOp(TensorDataset):
    def __init__(self, op: str = "x + y (mod 97)", ignore_idx: int = -100):
        self.op = OPERATIONS[op]["fn"]
        self.n_classes = OPERATIONS[op]["n_classes"]

        self.OP = self.n_classes
        self.EQ = self.n_classes + 1
        self.IG = ignore_idx

        self.vocab_size = self.n_classes + 2
        self.seq_len = 5

        x, y = [], []
        for a, b in product(range(self.n_classes), repeat=2):
            result = self.op(a, b)
            x.append([a, self.OP, b, self.EQ, result])
            y.append([self.IG, self.IG, self.IG, result, self.IG])
        x, y = torch.tensor(x), torch.tensor(y)
        super().__init__(x.to("cuda"), y.to("cuda"))

    def decode(self, indices: torch.Tensor) -> str:
        seq = []
        for idx in indices:
            match idx:
                case self.OP:
                    seq.append("+")
                case self.EQ:
                    seq.append("=")
                case self.IG:
                    seq.append("[M]")
                case _:
                    seq.append(str(idx.item()))
        return " ".join(seq)


if __name__ == "__main__":
    import random

    dataset = BinaryOp("x + y (mod 97)")

    for i in random.sample(range(len(dataset)), 10):
        x, y = dataset[i]
        print(dataset.decode(x))
