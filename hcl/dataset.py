import multiprocessing as mp
import os
from typing import Type, Union, Tuple
import logging

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import random_split, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

LOGGER = logging.getLogger(__name__)


class FashionMNISTPair(FashionMNIST):
    """Create paired dataset for contrastive learning"""
    def __getitem__(self, index):
        """Get two batches of images transformed randomly (therefore different)"""
        assert not self.target_transform, "target_transform has no effect"
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')  # consistent with all other datasets to return a PIL Image

        return self.transform(img), self.transform(img), target


class FashionMNISTDataModule(pl.LightningDataModule):
    """Solar dataset for classification"""
    def __init__(self,
                 ds: Type[FashionMNIST],
                 batch_size: int,
                 im_size: Union[Tuple[int, int], int] = 28,
                 data_dir: str = os.path.join(os.getcwd(), "FashionMNIST_data"),
                 num_workers: int = mp.cpu_count(),
                 splits: Tuple[int, int] = (57000, 2000, 1000),
                 supervised: bool = False):
        """
        Initialization
        :param data_dir: raw data folder (this dataset https://www.nature.com/articles/sdata2016106)
        :param batch_size: batch size
        :param im_size: image size
        :param num_workers: number of workers to load the data
        :param splits: dataset splits (unlabeled_train, labeled_train, val)
        :param supervised: if True, use only labels for training (supervised learning)
        """
        super().__init__()
        self.batch_size = batch_size
        self._num_workers = num_workers
        self._supervised = supervised

        # transforms
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(im_size, scale=(.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        # self.val_transforms = transforms.ToTensor()
        test_transforms = transforms.ToTensor()

        # create train-val-test dataset
        ds_train_val = ds(root=data_dir, transform=train_transforms, download=True)
        ds_train_unlabeled, ds_train_labels, ds_val = random_split(dataset=ds_train_val,
                                                                   lengths=splits,
                                                                   generator=torch.Generator().manual_seed(42))
        ds_test = ds(root=data_dir, transform=test_transforms, download=True, train=False)
        self.ds = {
            'train_unlabeled+labeled': ConcatDataset([ds_train_unlabeled, ds_train_labels]),
            'train_labeled': ds_train_labels,
            'val': ds_val,
            'test': ds_test
        }
        for key, val in self.ds.items():
            LOGGER.info(f"{key} datasets has {len(val)} data-points")

    def train_dataloader(self):
        """get training data loader"""
        name = 'train_labeled' if self._supervised else 'train_unlabeled+labeled'
        LOGGER.info(f'Train dataset is {name}')

        return DataLoader(self.ds[name], batch_size=self.batch_size, num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self):
        """get validation data loader"""
        return DataLoader(self.ds['val'], batch_size=self.batch_size, num_workers=self._num_workers)

    def test_dataloader(self):
        """get test data loader"""
        return DataLoader(self.ds['test'], batch_size=self.batch_size, num_workers=self._num_workers)
