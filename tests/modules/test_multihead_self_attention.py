import pytest
import torch as th
from conformer.modules import MultiHeadSelfAttention


def test_multihead_self_attention():
    mha = MultiHeadSelfAttention(embed_size=512, heads=8, mask=True)
    tensor = th.tensor(32, 128, 512)
    atten_out = mha(tensor)
    assert atten_out.shape = tensor.shape
