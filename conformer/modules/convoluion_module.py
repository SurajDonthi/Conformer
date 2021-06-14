import torch as th
import torch.nn.functional as F
from torch import nn


class ConvolutionModule(nn.Module):

    def __init__(self, input_dim: int,
                 expansion_factor: int = 4,
                 depth_kernel_size: int = 31,
                 drop_prob: int = 0.1):
        super().__init__()
        self._layers(input_dim, expansion_factor, depth_kernel_size, drop_prob)

    def _layers(self, input_dim: int,
                expansion_factor: int = 4,
                depth_kernel_size: int = 31,
                drop_prob: int = 0.1):
        self.layer_norm = nn.LayerNorm(input_dim)
        self.pointwise_conv1 = nn.Conv2d(input_dim,
                                         input_dim * expansion_factor,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0
                                         )
        self.depthwise_conv = nn.Conv1d(input_dim,
                                        input_dim * expansion_factor,
                                        kernel_size=depth_kernel_size,
                                        stride=1,
                                        padding=(depth_kernel_size - 1) // 2,
                                        bias=False
                                        )
        self.pointwise_conv2 = nn.Conv2d(input_dim,
                                         input_dim * expansion_factor,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0
                                         )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.layer_norm(x).transpose(1, 2).contiguous()
        x = F.glu(self.pointwise_conv1(x))
        x = F.silu(F.batch_norm(self.depthwise_conv(x)))

        x = self.dropout(self.pointwise_conv2(x))
        return x.transpose(1, 2).contiguous()
