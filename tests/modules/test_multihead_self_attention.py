import torch as th
from conformer.modules import MultiHeadSelfAttention


def test_input_with_default_values():
    mha = MultiHeadSelfAttention()
    tensor = th.rand(32, 128, 512)
    atten_out = mha(tensor)
    assert atten_out.shape == tensor.shape


def test_input_with_mask_true():
    mha = MultiHeadSelfAttention(embed_size=512, heads=8, mask=True)
    tensor = th.rand(32, 128, 512)
    atten_out = mha(tensor)
    assert atten_out.shape == tensor.shape
