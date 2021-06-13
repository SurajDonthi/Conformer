import csv
from argparse import Namespace
from pathlib import Path


def save_args(args: Namespace, save_dir: Path) -> None:
    with open(save_dir / 'hparams.csv', 'w') as f:
        csvw = csv.writer(f)
        csvw.writerow(['hparam', 'value'])
        for k, v in args.__dict__.items():
            csvw.writerow([k, v])
