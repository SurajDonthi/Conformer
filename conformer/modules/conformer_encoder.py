import torch as th
from torch import nn

from .conformer_block import ConformerBlock
from .convolution_module import Conv2dSubsampling


class ConformerEncoder(nn.Module):

    def __init__(self, input_dim: int = 80, encoder_dim: int = 512, num_layers: int = 5):
        super().__init__()
        self._layers(input_dim, encoder_dim, num_layers)

    def _layers(self, input_dim: int = 80, encoder_dim: int = 512, num_layers: int = 5):
        self.conv2d_subsampling = Conv2dSubsampling(1, encoder_dim)
        self.layers = nn.ModuleList(
            [ConformerBlock()] * num_layers)

    def forward(self, x):

        return x
