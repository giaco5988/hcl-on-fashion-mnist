import os
from typing import Tuple, Union, Type
import logging
import multiprocessing as mp

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as nn_func
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import numpy as np
from torch import nn
from fire import Fire

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
                 im_size: Union[Tuple[int, int], int] = 28,
                 data_dir: str = os.path.join(os.getcwd(), "FashionMNIST_data"),
                 batch_size=512,
                 num_workers: int = mp.cpu_count(),
                 train_val_split: Tuple[int, int] = (57000, 3000)):
        """
        Initialization
        :param data_dir: raw data folder (this dataset https://www.nature.com/articles/sdata2016106)
        :param im_size: image size
        :param batch_size: batch size
        :param num_workers: number of workers to load the data
        """
        super().__init__()
        self.batch_size = batch_size
        self._num_workers = num_workers

        # transforms
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(im_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        # self.val_transforms = transforms.ToTensor()
        test_transforms = transforms.ToTensor()

        # create train-val-test dataset
        ds_train_val = ds(root=data_dir, transform=train_transforms, download=True)
        assert sum(train_val_split) == len(ds_train_val), f"train-val splits should add up to {len(ds_train_val)}"
        ds_train, ds_val = random_split(dataset=ds_train_val, lengths=train_val_split)
        ds_test = ds(root=data_dir, transform=test_transforms, download=True, train=False)
        self.ds = {'train': ds_train, 'val': ds_val, 'test': ds_test}

    def train_dataloader(self):
        """get training data loader"""
        return DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self):
        """get validation data loader"""
        return DataLoader(self.ds['val'], batch_size=self.batch_size, num_workers=self._num_workers)

    def test_dataloader(self):
        """get test data loader"""
        return DataLoader(self.ds['test'], batch_size=self.batch_size, num_workers=self._num_workers)


class Model(pl.LightningModule):
    """"""
    def __init__(self, lr: float, tau_plus: float = .1, beta: float = 1.0, features: int = 512):
        """"""
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.tau_plus = tau_plus

        # load base model
        base = torchvision.models.resnet18(pretrained=False, progress=True)
        num_filters = base.fc.in_features

        # encoder
        self.encoder = []
        for name, module in base.named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)

        # projection head
        self.head = nn.Sequential(
            nn.Linear(num_filters, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Linear(512, features, bias=True)
        )

        # loss and metrics
        # self.metrics = MetricCollection([Accuracy(), Precision(), Recall()])
        self.loss = Model.criterion

        # save hyperparameters
        self.save_hyperparameters()

    @staticmethod
    def get_negative_mask(batch_size: int):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    @staticmethod
    def criterion(out_1, out_2, tau_plus, beta, estimator: str = 'hard', temperature: float = .5):
        """"""
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # old_neg = neg.clone()
        batch_size = out_1.shape[0]
        mask = Model.get_negative_mask(batch_size)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        elif estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss

    def forward(self, input_tensor: torch.Tensor):
        """Forward pass"""
        x = self.encoder(input_tensor)
        feature = torch.flatten(x, start_dim=1)
        out = self.head(feature)
        return nn_func.normalize(feature, dim=-1), nn_func.normalize(out, dim=-1)

    def training_step(self, batch, batch_idx):
        """training step"""
        # unpack batch
        im_1, im_2, _ = batch

        # compute loss
        _, out_1 = self(im_1)
        _, out_2 = self(im_2)
        loss = self.loss(out_1, out_2, self.tau_plus, self.beta)

        # log data
        # _, class_idx = torch.max(preds, 1)
        # metrics = {f'train_{key}': value for key, value in self.metrics(class_idx, y).items()}
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.cache_class["train"] += [(pred.item(), gt.item()) for pred, gt in zip(class_idx, y)]

        return loss

    def configure_optimizers(self):
        """configure optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        # lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, patience=10, factor=.5), "monitor": "train_loss"}

        return optimizer

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """training step"""
        # unpack batch
        im_1, im_2, _ = batch

        # compute loss
        _, out_1 = self(im_1)
        _, out_2 = self(im_2)
        loss = self.loss(out_1, out_2, self.tau_plus, self.beta)

        # log data
        self.log_dict({'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    # def test_step(self, batch, batch_idx) -> torch.Tensor:
    #     """It defines the test step"""
    #     im_1, _, label = batch
    #     preds = self(x)
    #     loss = self.test_loss(input=preds, target=y)
    #
    #     # accuracy
    #     _, class_idx = torch.max(preds, 1)
    #     metrics = {f'test_{key}': value for key, value in self.metrics(class_idx, y).items()}
    #     self.log_dict({'test_loss': loss, **metrics}, on_step=False, on_epoch=True, prog_bar=False, logger=False)
    #
    #     return loss


def finetune(logs_dir: str = os.getcwd(), max_epochs: int = 300) -> None:
    return NotImplementedError


class Cli:
    """Command line interface"""
    @staticmethod
    def train_hcl(logs_dir: str = os.getcwd(), max_epochs: int = 300) -> None:
        """
        Train binary classifier for solar panel detection
        :param logs_dir: where to save the logs
        :param max_epochs: max number of epochs
        :return: None
        """
        # set seed
        pl.seed_everything(1234)

        # initialize data and model
        ds = FashionMNISTDataModule(ds=FashionMNISTPair)
        model = Model(lr=1e-3)

        # initialize callbacks
        # callbacks = [
        #     EarlyStopping(monitor='train_loss', patience=20),
        #     LearningRateMonitor(logging_interval='step'),
        #     LogClassifierImages(batch_to_plot=(0,)),
        #     ModelCheckpoint(filename="max_val_acc_checkpoint", monitor="val_Accuracy", mode='max'),
        #     LogConfusionMatrix()
        # ]

        # initialize trainer and run it
        trainer = pl.Trainer(default_root_dir=logs_dir, gpus=torch.cuda.device_count(), max_epochs=max_epochs)
        trainer.fit(model, ds)


def testing():
    ds = FashionMNISTDataModule(ds=FashionMNISTPair, batch_size=512)

    for im, im2, label in ds.train_dataloader():
        print(im.shape, im2.shape, label.shape)

    for im, im2, label in ds.val_dataloader():
        print(im.shape, label.shape)

    for im, im2, label in ds.test_dataloader():
        print(im.shape, label.shape)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(Cli)
