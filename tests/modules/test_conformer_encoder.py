import pytest
import torch as th
from conformer.modules import ConformerEncoder
from loguru import logger


def test_input_with_default_values():
    batch_size, sequence_length, dim = 3, 12345, 80
    inputs = th.rand(batch_size, sequence_length, dim)
    input_lengths = th.IntTensor([12345, 12300, 12000])
    model = ConformerEncoder(num_layers=1)
    outputs, output_length = model(inputs, input_lengths)
    logger.info(
        f'Outputs shape: {outputs.shape}, Outputs length: {output_length}')
    assert outputs.shape == th.Size([3, 3085, 512])


@pytest.mark.skipif(not th.cuda.is_available(), reason='Skipping Test! GPU not Available!')
def test_input_with_default_values_with_gpu():
    batch_size, sequence_length, dim = 3, 12345, 80
    inputs = th.rand(batch_size, sequence_length, dim)
    input_lengths = th.IntTensor([12345, 12300, 12000])

    model = ConformerEncoder()
    outputs, output_lenth = model(inputs, input_lengths)
    assert outputs.shape == inputs.shape
