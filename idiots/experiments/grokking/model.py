import flax.linen as nn
import jax.numpy as jnp


class Block(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    old_parameterisation: bool = True  # For backwards compatibility

    @nn.compact
    def __call__(self, x):
        # Attention block
        h = nn.LayerNorm()(x)
        if self.old_parameterisation:
            q, k, v = jnp.split(nn.Dense(self.d_model * 3)(h), 3, axis=-1)
        else:
            # Turns out nn.MultiHeadAttention already does the Dense projection
            q, k, v = h, h, h
        casual_mask = nn.make_causal_mask(jnp.ones_like(x[:, :, 0]), dtype=jnp.bool_)
        h = nn.MultiHeadAttention(self.n_heads)(q, k, v, mask=casual_mask)
        x += h

        # MLP block
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d_ff)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        x += h

        return x


class TransformerSingleOutput(nn.Module):
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int
    max_len: int
    old_parameterisation: bool = True

    @nn.compact
    def __call__(self, x):
        # (b s) -> (b s d)
        tok_emb = nn.Embed(self.vocab_size, self.d_model)(x)
        pos_emb = nn.Embed(self.max_len, self.d_model)(jnp.arange(x.shape[1]))
        x = tok_emb + pos_emb

        for _ in range(self.n_layers):
            # (b s d) -> (b s d)
            x = Block(
                self.d_model, self.n_heads, self.d_model * 4, self.old_parameterisation
            )(x)

        # (b s d) -> (b v)
        logits = nn.Dense(self.vocab_size)(x[:, -1, :])
        return logits


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
