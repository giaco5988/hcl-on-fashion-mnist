from typing import Tuple, Optional
import logging

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as nn_func
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchmetrics import Accuracy, Precision, Recall, MetricCollection
import pytorch_lightning as pl
import numpy as np
from torch import nn
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(pl.LightningModule):
    """"""
    def __init__(self,
                 lr: float,
                 ds_memory: Optional[DataLoader] = None,
                 tau_plus: float = .1,
                 beta: float = 1.0,
                 features: int = 512,
                 temperature: float = .5,
                 num_classes: int = 10):
        """"""
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.tau_plus = tau_plus
        self.ds_memory = ds_memory
        self.feature_labels = None
        self.feature_bank = None
        self.temperature = temperature
        self.num_classes = num_classes
        self.k_closest = 100

        # load base model
        base = torchvision.models.resnet18(pretrained=False, progress=True)
        num_filters = base.fc.in_features

        # encoder
        self.encoder = []
        for name, module in base.named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1, bias=False)
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
        self.metrics = MetricCollection([Accuracy(), Precision(), Recall()])
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
    def criterion(out_1, out_2, tau_plus, beta, temperature: float, estimator: str = 'hard'):
        """"""
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # old_neg = neg.clone()
        batch_size = out_1.shape[0]
        mask = Model.get_negative_mask(batch_size).to(DEV)
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

    def on_validation_start(self) -> None:
        """Extract features for all labeled data in the training set (feature bank)"""
        feature_bank, targets = [], []
        for data, _, target in tqdm(self.ds_memory, desc='Feature extracting for bank', leave=False):
            feature, _ = self(data.to(DEV, non_blocking=True))
            feature_bank.append(feature)
            targets += list(target)
        self.feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        self.feature_labels = torch.tensor(targets, device=self.feature_bank.device)

    def on_validation_end(self) -> None:
        """Free variables"""
        self.feature_bank = None
        self.feature_labels = None

    def forward(self, input_tensor: torch.Tensor):
        """Forward pass"""
        x = self.encoder(input_tensor)
        feature = torch.flatten(x, start_dim=1)
        out = self.head(feature)
        return nn_func.normalize(feature, dim=-1), nn_func.normalize(out, dim=-1)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """training step"""
        # unpack batch
        im_1, im_2, _ = batch

        # compute loss
        _, out_1 = self(im_1)
        _, out_2 = self(im_2)
        loss = self.loss(out_1, out_2, self.tau_plus, self.beta, self.temperature)

        # log data
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        """configure optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, patience=10, factor=.5), "monitor": "train_loss"}

        return [optimizer], [lr_scheduler]

    def guess_label(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Guess label based on embeddings similarity"""
        # unpack batch
        data, data_2, target = batch

        # compute loss
        feature, out = self(data)
        _, out_2 = self(data_2)
        loss = self.loss(out, out_2, self.tau_plus, self.beta, temperature=self.temperature)

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, self.feature_bank)  # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.k_closest, dim=-1)  # [B, K]
        sim_labels = torch.gather(self.feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.temperature).exp()

        # counts for each class
        one_hot_label = torch.zeros(data.size(0) * self.k_closest, self.num_classes, device=sim_labels.device)  # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)  # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, self.num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        return loss, pred_scores, target

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """training step"""
        loss, pred_scores, targets = self.guess_label(batch=batch)

        # log data
        _, class_idx = torch.max(pred_scores, 1)
        metrics = {f'val_{key}': value for key, value in self.metrics(class_idx, targets).items()}
        self.log_dict({'val_loss': loss, **metrics}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """test step"""
        loss, pred_scores, targets = self.guess_label(batch=batch)

        # log data
        _, class_idx = torch.max(pred_scores, 1)
        metrics = {f'test_{key}': value for key, value in self.metrics(class_idx, targets).items()}
        self.log_dict({'test_loss': loss, **metrics}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss


class Finetuner(pl.LightningModule):
    """"""
    def __init__(self, lr: float, pretrained_path: str, num_classes: int = 10, train_encoder: bool = False):
        """"""
        super().__init__()
        self.lr = lr

        # load base model
        model = Model(lr=self.lr)
        model.load_state_dict(torch.load(pretrained_path, map_location=DEV))
        for param in model.encoder.parameters():
            param.requires_grad = train_encoder

        # encoder, classifier, loss and metrics
        self.encoder = model.encoder
        self.fc = nn.Linear(model.head[0].in_features, num_classes, bias=True)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = MetricCollection([Accuracy(), Precision(), Recall()])

        # save hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        """configure optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, patience=10, factor=.5), "monitor": "train_loss"}

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        """forward pass"""
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)

        return out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """training step"""
        im, _, target = batch  # unpack batch
        pred = self(im)  # inference
        loss = self.loss(input=pred, target=target)  # compute loss

        # log data
        _, class_idx = torch.max(pred, 1)
        metrics = {f'train_{key}': value for key, value in self.metrics(class_idx, target).items()}
        self.log_dict({'train_loss': loss, **metrics}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """training step"""
        im, _, target = batch  # unpack batch
        pred = self(im)  # inference
        loss = self.loss(input=pred, target=target)  # compute loss

        # log data
        _, class_idx = torch.max(pred, 1)
        metrics = {f'val_{key}': value for key, value in self.metrics(class_idx, target).items()}
        self.log_dict({'val_loss': loss, **metrics}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """training step"""
        im, _, target = batch  # unpack batch
        pred = self(im)  # inference
        loss = self.loss(input=pred, target=target)  # compute loss

        # log data
        _, class_idx = torch.max(pred, 1)
        metrics = {f'test_{key}': value for key, value in self.metrics(class_idx, target).items()}
        self.log_dict({'test_loss': loss, **metrics}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss
