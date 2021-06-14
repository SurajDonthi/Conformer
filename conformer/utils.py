import csv
from argparse import Namespace
from pathlib import Path

import torch as th


def save_args(args: Namespace, save_dir: Path) -> None:
    with open(save_dir / 'hparams.csv', 'w') as f:
        csvw = csv.writer(f)
        csvw.writerow(['hparam', 'value'])
        for k, v in args.__dict__.items():
            csvw.writerow([k, v])


def mask_(tensors: th.Tensor, maskval: float = 0.0, mask_diagonal: bool = True):
    """
        Description:
        Masks out all values in the given batch of tensors where i <= j holds,
        i < j if mask_diagonal is false
        In place operation

    Args:
        tensors (Tensor): Batch of Tensors
        maskval (float, optional): Mask value to set. Defaults to 0.0.
        mask_diagonal (bool, optional): Whether to also mask the diagonal. Defaults to True.
    """
    h, w = tensors.size(-2), tensors.size(-1)

    indices = th.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    tensors[..., indices[0], indices[1]] = maskval
