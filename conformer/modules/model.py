from typing import Literal

import torch as th
from torch import nn

from .conformer_encoder import ConformerEncoder
from .linear import Linear
from .rnnt_decoder import RNNTDecoder


class ConformerModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 80,
        encoder_dim: int = 512,
        decoder_dim: int = 640,
        num_attention_heads: int = 8,
        depth_conv_kernel_size: int = 31,
        num_conformer_layers: int = 17,
        num_decoder_rnn_layers: int = 1,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        rnn_type: Literal['lstm', 'gru', 'rnn'] = 'lstm',
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        decoder_drop_p: float = 0.1
    ):
        super().__init__()

        self.encoder = ConformerEncoder(
            input_dim,
            encoder_dim,
            num_conformer_layers,
            num_attention_heads,
            feed_forward_expansion_factor,
            conv_expansion_factor,
            input_dropout_p,
            feed_forward_dropout_p,
            attention_dropout_p,
            conv_dropout_p,
            depth_conv_kernel_size
        )
        self.decoder = RNNTDecoder(
            num_classes,
            decoder_dim,
            encoder_dim,
            num_decoder_rnn_layers,
            rnn_type,
            dropout_p=decoder_drop_p
        )
        self.joint = JointNetwork()
        self.fc = Linear(encoder_dim << 1, num_classes, bias=False)

    def forward(self, input, input_length, target, target_length):
        encoder_out, _ = self.encoder(input, input_length)
        decoder_out, _ = self.decoder(target, target_length)
        output = self.joint(encoder_out, decoder_out)
        output = self.fc(output)
        return output


class JointNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, decoder_outputs):
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = th.cat((encoder_outputs, decoder_outputs), dim=-1)
        return outputs
