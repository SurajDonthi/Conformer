import torch as th
from torch import nn

from .conformer_block import ConformerBlock
from .convolution_module import Conv2dSubsampling


class ConformerEncoder(nn.Module):

	def __init__(self, ):
		super().__init__()
		self._layers()

	def _layers(self, ):
		self.conv2d_subsampling = Conv2dSubsampling(1, encoder_dim)
		self.layers = nn.ModuleList(
			[ConformerBlock()] * num_layers)

	def forward(self, x):
		
		return x
