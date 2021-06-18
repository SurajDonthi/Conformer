from torch.utils.data import DataLoader, Dataset

from base import BaseDataModule


class SpeechDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x


class CustomDataLoader(BaseDataModule):

    def __init__(self, data_dir: str,
                 train_batchsize: int = 32,
                 val_batchsize: int = 32,
                 test_batchsize: int = 32,
                 num_workers: int = 4,
                 ):

        super().__init__()

        self.data_dir = data_dir

        if not self.data_dir.exists():
            raise Exception(
                f"'Path '{self.data_dir.__str__()}' does not exist!")
        if not self.data_dir.is_dir():
            raise Exception(
                f"Path '{self.data_dir.__str__()}' is not a directory!")

        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.val_batchsize = val_batchsize
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        return super().prepare_data(*args, **kwargs)

    def setup(self, stage=None):
        return super().setup(stage=stage)

    def train_dataloader(self):
        dataset = Dataset()
        return DataLoader(dataset, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = Dataset()
        return DataLoader(dataset, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        dataset = Dataset()
        return DataLoader(dataset, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers)
