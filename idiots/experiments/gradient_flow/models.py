import flax.linen as nn
import jax.numpy as jnp

from idiots.experiments.classification.model import ImageMLP


class EmbedMLP(nn.Module):
    hidden: int
    n_layers: int
    n_classes: int

    @nn.compact
    def __call__(self, x):
        b, s = x.shape
        tok_emb = nn.Embed(self.n_classes, self.hidden)(x)
        pos_emb = nn.Embed(s, self.hidden)(jnp.arange(s))
        x = tok_emb + pos_emb
        x = x.reshape(b, -1)

        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_classes)(x)
        return x
