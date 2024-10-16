import pytorch_lightning as L

from typing import Self
from datasets import load_dataset
from torch.utils.data import DataLoader


class Minipile(L.LightningDataModule):
    def __init__(self: Self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self: Self) -> None:
        self.dataset = load_dataset("JeanKaddour/minipile")

    def setup(self: Self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset["train"]
            self.val = self.dataset["validation"]
        if stage == "test" or stage is None:
            self.test = self.dataset["test"]

    def train_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size)
