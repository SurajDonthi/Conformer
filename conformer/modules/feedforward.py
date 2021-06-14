import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, input_dim: int = 512,
                 expansion_factor: int = 4,
                 drop_prob: int = 0.1):
        super().__init__()
        self._layers(input_dim, expansion_factor, drop_prob)

    def _layers(self, input_dim, expansion_factor, drop_prob):
        self.linear1 = nn.Linear(input_dim, expansion_factor)
        self.dropout1 = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(expansion_factor, input_dim)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x):
        # Swish Activation Function
        x = self.dropout1(F.silu(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x
