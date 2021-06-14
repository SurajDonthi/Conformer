import torch as th
from conformer.modules import FeedForward


def test_input_with_default_values():
    ff = FeedForward()
    tensor = th.rand(32, 128, 512)
    atten_out = ff(tensor)
    assert atten_out.shape == tensor.shape
