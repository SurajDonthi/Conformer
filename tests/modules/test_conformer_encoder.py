import torch as th
from conformer.modules import ConformerEncoder


def test_input_with_default_values():
    model = ConformerEncoder()
    tensor = th.rand(32, 128, 512)
    atten_out = model(tensor)
    assert atten_out.shape == tensor.shape
