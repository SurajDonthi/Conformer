from argparse import ArgumentParser
from typing import Literal, Union

import torch.nn.functional as F
from pytorch_lightning.utilities import parsing
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (LambdaLR, MultiStepLR, ReduceLROnPlateau,
                                      StepLR, _LRScheduler)

from .base import BaseModule
from .modules import ConformerModel

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss,
          'ctc': F.ctc_loss
          }

OPTIMIZERS = {
    'sgd': SGD,
    'adamw': AdamW,
    'adam': Adam
}

SCHEDULERS = {
    'step_lr': StepLR,
    'lambda_lr': LambdaLR,
    'multistep_lr': MultiStepLR,
    'reduce_lr_on_plateau': ReduceLROnPlateau
}


class Pipeline(BaseModule):

    def __init__(
        self,
        model: nn.Module,
        model_args: dict = {},
        criterion: Union[Literal[tuple(LOSSES.keys())], _Loss] = 'ctc',
        optim: Union[Literal[tuple(OPTIMIZERS.keys())], Optimizer] = 'adamw',
        optim_args: dict = {},
        lr=0.0001,
        scheduler: Union[Literal[tuple(SCHEDULERS.keys())], _LRScheduler] = 'reduce_lr_on_plateau',
        schedular_args: dict = {}
    ):
    super().__init__()

    self.model = model(**model_args)
    lr = self.lr
    self.criterion = criterion
    self.optim = optim(**optim_args)
    self.scheduler = scheduler(**schedular_args)

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
        return self.optim, self.scheduler

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self(inputs)

        loss = self.criterion(preds, targets)
        self.log(loss, prog_bar=True)

    def validation_step(self, batch, bathc_idx):

        loss = None
        self.log(loss, progress_bar=True)

    def test_step(self, batch, bathc_idx):

        loss = None
        self.log(loss, progress_bar=True)
