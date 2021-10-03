import os
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from fire import Fire

from hcl.model import Model, Finetuner
from hcl.dataset import FashionMNISTPair, FashionMNISTDataModule

LOGGER = logging.getLogger(__name__)
CALLBACKS = [
    EarlyStopping(monitor='train_loss', patience=20),
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(filename="max_val_acc_checkpoint", monitor="val_Accuracy", mode='max')
]


class Cli:
    """Command line interface"""
    @staticmethod
    def train_hcl(logs_dir: str = os.getcwd(), max_epochs: int = 100) -> None:
        """
        Train binary classifier for solar panel detection
        :param logs_dir: where to save the logs
        :param max_epochs: max number of epochs
        :return: None
        """
        # initialize data and model
        ds = FashionMNISTDataModule(ds=FashionMNISTPair)
        model = Model(lr=1e-3, ds_memory=DataLoader(ds.ds['train_labeled'], batch_size=16))

        # initialize trainer and run it
        gpus = torch.cuda.device_count()
        trainer = pl.Trainer(default_root_dir=logs_dir, gpus=gpus, callbacks=CALLBACKS, max_epochs=max_epochs)
        trainer.fit(model, ds)

    @staticmethod
    def finetune(pretrained_path: str, logs_dir: str = os.getcwd(), max_epochs: int = 300) -> None:
        """
        Train binary classifier for solar panel detection
        :param pretrained_path: where to load pretrained model (relative to cwd)
        :param logs_dir: where to save the logs
        :param max_epochs: max number of epochs
        :return: None
        """
        # initialize data and model
        ds = FashionMNISTDataModule(ds=FashionMNISTPair)
        model = Finetuner(lr=1e-3, pretrained_path=os.path.join(os.getcwd(), pretrained_path))

        # initialize callbacks
        callbacks = [
            EarlyStopping(monitor='train_loss', patience=20),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(filename="max_val_acc_checkpoint", monitor="val_Accuracy", mode='max')
        ]

        # initialize trainer and run it
        gpus = torch.cuda.device_count()
        trainer = pl.Trainer(default_root_dir=logs_dir, gpus=gpus, callbacks=callbacks, max_epochs=max_epochs)
        trainer.fit(model, ds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(Cli)