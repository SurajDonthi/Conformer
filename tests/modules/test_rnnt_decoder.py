import pytest
import torch as th
from conformer.modules import RNNTDecoder
from loguru import logger


def test_input_with_default_values():
    num_classes, hidden_state_dim, output_dim, num_layers, drop_p = 10, 640, 512, 1, 0.1
    targets = th.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                             [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                             [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]])
    target_lengths = th.LongTensor([9, 8, 7])
    model = RNNTDecoder(num_classes, hidden_state_dim, num_layers, output_dim,
                        dropout_p=drop_p)
    outputs, output_length = model(targets, target_lengths)
    logger.info(
        f'Outputs shape: {outputs.shape}, num_output_lengths: {len(output_length)}, '
        f'Outputs length: {output_length[0].shape}')
    assert outputs.shape == th.Size([3, 9, 512])
    # assert output_length == th.Tensor([3085, ])
