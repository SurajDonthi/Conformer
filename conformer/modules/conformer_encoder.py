from typing import Tuple

import torch as th
from torch import nn

from .conformer_block import ConformerBlock
from .convolution_module import Conv2dSubsampling
from .linear import Linear


class ConformerEncoder(nn.Module):

    def __init__(self, input_dim: int = 80,
                 encoder_dim: int = 512,
                 num_layers: int = 5,
                 input_dropout_p: float = 0.1):
        super().__init__()
        self._layers(input_dim, encoder_dim, num_layers, input_dropout_p)

    def _layers(self, input_dim: int = 80,
                encoder_dim: int = 512,
                num_layers: int = 5,
                input_dropout_p: float = 0.1):
        self.conv2d_subsampling = Conv2dSubsampling(1, encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock()] * num_layers)

    def forward(self, x: th.Tensor, input_lengths: int) -> Tuple[th.Tensor, th.Tensor]:
        outputs, output_lengths = self.conv2d_subsampling(x, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.conformer_layers:
            outputs = layer(outputs)

        return outputs, output_lengths
