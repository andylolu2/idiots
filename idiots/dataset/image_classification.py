from datasets import DatasetDict, load_dataset


def mnist_splits():
    """MNIST dataset.

    Features:
        x: (28, 28) uint8
        y: () int32
    """
    ds_dict = load_dataset("mnist")
    assert isinstance(ds_dict, DatasetDict)

    ds_dict = ds_dict.rename_columns({"image": "x", "label": "y"}).with_format("jax")

    return ds_dict["train"], ds_dict["test"]


if __name__ == "__main__":
    from idiots.dataset.dataloader import DataLoader

    ds_train, ds_test = mnist_splits()
    print(ds_train.features)
    for item in DataLoader(ds_train, 32):
        print(item["x"].shape, item["y"].shape)
        print(item["x"].dtype, item["y"].dtype)
        break
