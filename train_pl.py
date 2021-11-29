import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import glob
import json
import numpy as np
import torchmetrics
from options.pl_options import PLOptions
from data import DataLoader
from models import create_model
from models.losses import ce_jaccard


class MeshSegmenter(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = create_model(opt)
        self.criterion = ce_jaccard
        if self.training:
            self.train_metrics = torch.nn.ModuleList([
                torchmetrics.Accuracy(num_classes=opt.nclasses, average='macro'),
                torchmetrics.IoU(num_classes=opt.nclasses),
                torchmetrics.F1(num_classes=opt.nclasses, average='macro')
            ])
            self.val_metrics = torch.nn.ModuleList([
                torchmetrics.Accuracy(num_classes=opt.nclasses, average='macro'),
                torchmetrics.IoU(num_classes=opt.nclasses),
                torchmetrics.F1(num_classes=opt.nclasses, average='macro')
            ])

    def training_step(self, batch, idx):
        self.model.set_input(batch)
        out = self.model.forward()
        loss = self.criterion(self.model.labels, out)

        pred_class = out.data.max(1)[1]
        not_padding = self.model.labels != -1
        label_class = self.model.labels[not_padding]
        pred_class = pred_class[not_padding]

        for m in self.train_metrics:
            val = m(pred_class, label_class)
            metric_name = str(m).split('(')[0]
            self.log(metric_name.lower(), val, logger=True)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, idx):
        self.model.set_input(batch)
        out = self.model.forward()
        loss = self.criterion(self.model.labels, out)

        pred_class = out.data.max(1)[1]
        not_padding = self.model.labels != -1
        label_class = self.model.labels[not_padding]
        pred_class = pred_class[not_padding]

        for m in self.val_metrics:
            val = m(pred_class, label_class)
            metric_name = str(m).split('(')[0]
            self.log('val_' + metric_name.lower(), val, logger=True)
        self.log('val_loss', loss)
        return loss

    def forward(self, image):
        return self.model(image)

    def on_train_epoch_end(self, unused = None):
        for m in self.train_metrics:
            m.reset()

    def on_validation_epoch_end(self) -> None:
        for m in self.val_metrics:
            m.reset()

    def train_dataloader(self):
        self.opt.phase = 'train'
        return DataLoader(self.opt)

    def val_dataloader(self):
        self.opt.phase = 'test'
        return DataLoader(self.opt)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.model.net.parameters(), lr=self.opt.lr,
                              momentum=0.9,
                              weight_decay=0.0002)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.opt.max_epochs * 2)
        return [opt], [sched]


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=60)

    parser.add_argument('--progress_bar_refresh_rate', type=int, default=20)
    parser.add_argument('--default_root_dir', default='checkpoints/', help='pytorch-lightning log path')
    return parser


if __name__ == '__main__':
    from pytorch_lightning.callbacks import ModelCheckpoint
    args = PLOptions().parse()
    model = MeshSegmenter(args)
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[ModelCheckpoint(monitor='val_iou',
                                                                       mode='max',
                                                                       save_top_k=3,
                                                                       filename='{epoch:02d}-{val_acc_epoch:.2f}',)])
    trainer.fit(model)