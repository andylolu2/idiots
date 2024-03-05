import flax.linen as nn
import jax.numpy as jnp


class ImageMLP(nn.Module):
    hidden: int
    n_layers: int
    out: int
    normalize_inputs: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        if self.normalize_inputs:
            x = x.astype(jnp.float32) / 255.0
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out)(x)
        return x
