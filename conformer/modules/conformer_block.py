from torch import nn

from .convolution_module import ConvolutionModule
from .feedforward import FeedForward
from .multihead_self_attention import MultiHeadSelfAttention


class ConformerBlock(nn.Module):

    def __init__(self, embed_size: int = 512,
                 heads: int = 8,
                 expansion_factor: int = 4,
                 depth_kernel_size: int = 31,
                 drop_prob: float = 0.1):
        super().__init__()
        self._layers(embed_size, heads, expansion_factor, depth_kernel_size, drop_prob)

    def _layers(self, embed_size: int, heads: int, expansion_factor: int,
                depth_kernel_size: int, drop_prob: float):
        self.ff1 = FeedForward(embed_size, expansion_factor, drop_prob)
        self.mha = MultiHeadSelfAttention(embed_size, heads=heads)
        self.conv = ConvolutionModule(embed_size,
                                      expansion_factor,
                                      depth_kernel_size,
                                      drop_prob=drop_prob,
                                      )
        self.ff2 = FeedForward(embed_size, expansion_factor, drop_prob)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = self.ff1(x) + x
        x = self.mha(x) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        return self.layer_norm(x)
