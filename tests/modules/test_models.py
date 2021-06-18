import pytest
import torch as th
from conformer.modules import ConformerModel
from loguru import logger


def test_input_with_default_values():
    batch_size, sequence_length, dim = 3, 12345, 80
    inputs = th.rand(batch_size, sequence_length, dim)
    input_lengths = th.IntTensor([12345, 12300, 12000])

    num_classes = 10
    # num_classes, hidden_state_dim, output_dim, num_layers, drop_p = 10, 640, 512, 1, 0.1

    num_conformer_layers = 1

    targets = th.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                             [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                             [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]])
    target_lengths = th.LongTensor([9, 8, 7])

    model = ConformerModel(
        num_classes=num_classes,
        num_conformer_layers=num_conformer_layers
    )

    outputs, output_length = model(inputs, input_lengths, targets, target_lengths)
    logger.info(
        f'Outputs shape: {outputs.shape}, Outputs length: {output_length}')
    assert outputs.shape == th.Size([3, 9, 512])
