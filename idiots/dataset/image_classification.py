from datasets import DatasetDict, load_dataset


def mnist_splits(train_size: int, test_size: int, seed: int = 0):
    """MNIST dataset.

    Features:
        x: (28, 28) uint8
        y: () int32
    """
    ds_dict = load_dataset("mnist", keep_in_memory=True)
    assert isinstance(ds_dict, DatasetDict)

    ds_dict = ds_dict.rename_columns({"image": "x", "label": "y"}).with_format("numpy")
    ds_train, ds_test = ds_dict["train"], ds_dict["test"]
    ds_train = ds_train.shuffle(seed=seed).select(range(train_size))
    ds_test = ds_test.shuffle(seed=seed + 1).select(range(test_size))

    return ds_train, ds_test


if __name__ == "__main__":
    from time import perf_counter

    from idiots.dataset.dataloader import DataLoader

    ds_train, ds_test = mnist_splits(50000, 10000)
    print(ds_train.features)
    for item in DataLoader(ds_train, 32):
        print(item["x"].shape, item["y"].shape)
        print(item["x"].dtype, item["y"].dtype)
        break

    # Performance test
    start = perf_counter()
    n_iters = 0
    for batch in DataLoader(ds_train, 256):
        n_iters += 1
    print(f"Time: {perf_counter() - start:.4f} seconds")
    print(f"Time/step: {(perf_counter() - start) / 100:.4f} seconds")
