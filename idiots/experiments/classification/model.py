import flax.linen as nn


class ImageMLP(nn.Module):
    hidden: int
    n_layers: int
    out: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out)(x)
        return x
