from typing import Tuple

import torch as th
from torch import nn
from typing_extensions import Literal

from .linear import Linear


class RNNTDecoder(nn.Module):
    rnn_type = {
        'lstm': nn.LSTM,
        'rnn': nn.RNN,
        'gru': nn.GRU
    }

    def __init__(
            self,
            num_classes: int,
            hidden_state_dim: int,
            num_layers: int,
            output_dim: int,
            rnn_type: Literal['lstm', 'gru', 'rnn'] = 'lstm',
            sos_id: int = 1,
            eos_id: int = 2,
            dropout_p: float = 0.2):
        super().__init__()

        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.rnn = self.rnn_type[rnn_type](
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False
        )
        self.linear = Linear(hidden_state_dim, output_dim)

    def forward(
            self,
            inputs: th.Tensor,
            input_lengths: th.Tensor = None,
            hidden_states: th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        embedded = self.embedding(inputs)

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded.transpose(0, 1), input_lengths.cpu())
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = self.linear(outputs.transpose(0, 1))
        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs = self.linear(outputs)

        return outputs, hidden_states
