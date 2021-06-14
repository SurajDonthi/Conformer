from argparse import ArgumentParser

import torch.nn.functional as F
from pytorch_lightning.utilities import parsing

from base import BaseModule
from models import Model

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}


class Pipeline(BaseModule):

    def __init__(self, lr=0.0001, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-des', '--description', required=False, type=str)
        parser.add_argument('-lp', '--log_path', type=str,
                            default='./logs')
        parser.add_argument('-gt', '--git_tag', type=parsing.str_to_bool,
                            default=False, help='Creates a git tag if true')
        parser.add_argument('--debug', type=parsing.str_to_bool,
                            default=False, help='Does not log if debug mode is true')
        return parser

    def configure_optimizers(self):
        optim = None
        scheduler = None
        return optim, scheduler

    def training_step(self, X):

        loss = None
        self.log(loss, prog_bar=True)

    def validation_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)

    def test_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)
