from torch import nn
from x_transformers import Decoder, TransformerWrapper

from idiots.utils import num_parameters


class TransformerSingleOutput(nn.Module):
    def __init__(
        self, num_tokens: int, max_seq_len: int, dim: int, depth: int, heads: int
    ):
        super().__init__()
        self.model = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(dim=dim, depth=depth, heads=heads),
        )

        print(f"Number of parameters: {num_parameters(self):,}")

    def forward(self, x):
        logits = self.model(x)
        return logits[:, -1, :].clone()  # (b s d) -> (b d)
