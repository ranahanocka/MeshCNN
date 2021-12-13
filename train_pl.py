import argparse
import os
import random

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
from models.losses import postprocess
from models.losses import ce_jaccard
import warnings
warnings.filterwarnings("ignore")


class MeshSegmenter(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = create_model(opt)
        if opt.from_pretrained is not None:
            print('Loaded pretrained weights:', opt.from_pretrained)
            self.model.load_weights(opt.from_pretrained)
        self.criterion = self.model.criterion
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

    def step(self, batch, metrics, metric_prefix=''):
        self.model.set_input(batch)
        out = self.model.forward()
        true, pred = postprocess(self.model.labels, out)
        loss = self.criterion(true, pred)

        true = true.view(-1)
        pred = pred.argmax(1).view(-1)

        prefix = metric_prefix
        for m in metrics:
            val = m(pred, true)
            metric_name = str(m).split('(')[0]
            self.log(prefix + metric_name.lower(), val, logger=True, prog_bar=True, on_epoch=True)
        self.log(prefix + 'loss', loss, on_epoch=True)
        return loss

    def training_step(self, batch, idx):

        return self.step(batch, self.train_metrics)

    def validation_step(self, batch, idx):
        return self.step(batch, self.val_metrics, metric_prefix='val_')

    def forward(self, image):
        return self.model(image)

    def on_train_epoch_end(self, unused=None):
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
        # opt = torch.optim.Adam(self.model.net.parameters(), lr=self.opt.lr, weight_decay=0.0002)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.opt.max_epochs * 2)
        return [opt], [sched]


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