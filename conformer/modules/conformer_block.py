from torch import nn

from .convolution_module import ConvolutionModule
from .feedforward import FeedForward
from .multihead_self_attention import MultiHeadSelfAttention


class ConformerBlock(nn.Module):

    def __init__(
        self,
        embed_size: int = 512,
        heads: int = 8,
        expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        depth_kernel_size: int = 31,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
    ):
        super().__init__()

        self.ff1 = FeedForward(embed_size, expansion_factor, feed_forward_dropout_p)
        self.mha = MultiHeadSelfAttention(embed_size, heads=heads)
        self.conv = ConvolutionModule(embed_size,
                                      conv_expansion_factor,
                                      depth_kernel_size,
                                      drop_prob=attention_dropout_p,
                                      )
        self.ff2 = FeedForward(embed_size, expansion_factor, conv_dropout_p)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = self.ff1(x) / 2 + x
        x = self.mha(x) + x
        x = self.conv(x) + x
        x = self.ff2(x) / 2 + x
        return self.layer_norm(x)
