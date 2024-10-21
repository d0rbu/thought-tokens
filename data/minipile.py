import pytorch_lightning as L

from typing import Self, Callable
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset


class Minipile(L.LightningDataModule):
    def __init__(
        self: Self,
        init_size: int = 7,
        transform: Callable | None = None,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 8
    ) -> None:
        super().__init__()
        self.init_size = init_size
        self.transform = transform
        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }

    def prepare_data(self: Self) -> None:
        self.dataset = load_dataset("JeanKaddour/minipile")

        if self.transform is not None:
            self.dataset = self.dataset.map(self.transform, batched=True)

    def setup(self: Self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset["train"]
            self.val = self.dataset["validation"]
            self.init = Subset(self.train, range(self.init_size))
        if stage == "test" or stage is None:
            self.test = self.dataset["test"]

    def init_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.init, **self.dataloader_params)

    def train_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.train, **self.dataloader_params)

    def val_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.val, **self.dataloader_params)

    def test_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.test, **self.dataloader_params)
