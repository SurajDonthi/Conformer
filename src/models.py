import torch as th
import torch.nn.functional as F
from torch import nn

from utils import mask_


class MultiHeadSelfAttention(nn.Module):
    """
    Description:
        Canonical implementation of multi-head self attention.
        Source: https://github.com/pbloem/former
    """

    def __init__(self, embed_size: int, heads: int = 8, mask: bool = False):
        super().__init__()
        assert embed_size % heads == 0, f'Embedding dimension ({embed_size}) should be " \
            "divisible by nr. of heads ({heads})'
        self._layers(embed_size, heads, mask)

    def _layers(self, embed_size: int, heads: int, mask: bool = False):
        self.embed_size = embed_size
        self.heads = heads
        self.mask = mask

        # sub_size = embed_size // heads
        # - We will break the embedding into `heads` chunks and feed each to a different
        # attention head

        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.merged_heads = nn.Linear(embed_size, embed_size)

    def forward(self, x: th.Tensor) -> th.tensor:
        batch_size, seq_len, embed_size = x.size()
        sub_size = embed_size // self.heads
        assert embed_size == self.embed_size, f'Input embedding dim ({embed_size}) should " \
            "match layer embedding dim ({self.embed_size})'

        keys = self.keys(x).view(batch_size, seq_len, self.heads, sub_size)
        queries = self.queries(x).view(batch_size, seq_len, self.heads, sub_size)
        values = self.values(x).view(batch_size, seq_len, self.heads, sub_size)

        # -- We first compute the k/q/v'sub_size on the whole embedding vectors, and
        # then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(
            batch_size * self.heads, seq_len, sub_size)
        queries = queries.transpose(1, 2).contiguous().view(
            batch_size * self.heads, seq_len, sub_size)
        values = values.transpose(1, 2).contiguous().view(
            batch_size * self.heads, seq_len, sub_size)

        queries = queries / (embed_size ** (1/4))
        keys = keys / (embed_size ** (1/4))
        # - Instead of dividing the dot products by sqrt(embed_size), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = th.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (batch_size*self.heads, seq_len, seq_len)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = th.bmm(dot, values).view(batch_size, self.heads, seq_len, sub_size)

        # swap self.heads, seq_len back, unify heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, sub_size * self.heads)

        return self.merged_heads(out)


class ResidulLayerNorm(nn.Module):

    def __init__(self, dim, p):
        super().__init__()
        self._layers(dim, p)

    def _layers(self, dim, p):
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p)

    def forward(self, *tensors: th.Tensor):
        return self.layer_norm(self.dropout(*tensors) + tensors[-1])
