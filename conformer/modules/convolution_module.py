import torch as th
import torch.nn.functional as F
from torch import nn


class ConvolutionModule(nn.Module):

    def __init__(self, input_dim: int = 512,
                 expansion_factor: int = 2,
                 depth_kernel_size: int = 31,
                 drop_prob: int = 0.1):
        super().__init__()
        self._layers(input_dim, expansion_factor, depth_kernel_size, drop_prob)

    def _layers(self, input_dim: int,
                expansion_factor: int = 2,
                depth_kernel_size: int = 31,
                drop_prob: int = 0.1):
        self.layer_norm = nn.LayerNorm(input_dim)
        self.pointwise_conv1 = nn.Conv1d(input_dim,
                                         input_dim * expansion_factor,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0
                                         )
        self.depthwise_conv = nn.Conv1d(input_dim,
                                        input_dim,
                                        kernel_size=depth_kernel_size,
                                        stride=1,
                                        padding=(depth_kernel_size - 1) // 2,
                                        bias=False
                                        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.pointwise_conv2 = nn.Conv1d(input_dim,
                                         input_dim,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0
                                         )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.layer_norm(x).transpose(1, 2).contiguous()
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = F.silu(self.batch_norm(self.depthwise_conv(x)))

        x = self.dropout(self.pointwise_conv2(x))
        return x.transpose(1, 2).contiguous()


class Conv2dSubsampling(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._layers(in_channels, out_channels)

    def _layers(self, in_channels: int, out_channels: int):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x: th.Tensor, input_lengths: int):
        x = th.relu(self.conv1(x))
        x = th.relu(self.conv2(x))
        batch_size, channels, subsampled_lengths, subsampled_dim = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)
        output_lengths = input_lengths >> 2
        output_lengths -= 1
        return x, output_lengths
