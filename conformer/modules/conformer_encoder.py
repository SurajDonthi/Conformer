from typing import Tuple

import torch as th
from torch import nn

from .conformer_block import ConformerBlock
from .convolution_module import Conv2dSubsampling
from .linear import Linear


class ConformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 5,
        heads: int = 8,
        expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        depth_conv_kernel_size: int = 31
    ):
        super().__init__()

        self.conv2d_subsampling = Conv2dSubsampling(1, encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                encoder_dim,
                heads,
                expansion_factor,
                conv_expansion_factor,
                depth_conv_kernel_size,
                feed_forward_dropout_p,
                attention_dropout_p,
                conv_dropout_p,
            )] * num_layers)

    def forward(self, x: th.Tensor, input_lengths: int) -> Tuple[th.Tensor, th.Tensor]:
        outputs, output_lengths = self.conv2d_subsampling(x, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.conformer_layers:
            outputs = layer(outputs)

        return outputs, output_lengths
