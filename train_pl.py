import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms as T
from torchvision import transforms
import json
import numpy as np
import imgaug as ia
from utils.image_processing import resize_image, pad_image
import torchmodels
import torchmetrics
from utils.image_processing import enhance_contrast
from imgaug import augmenters as iaa
import pandas as pd
from options.pl_options import PLOptions


class MeshSegmenter(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = create_model(opt)
        self.train_metrics = [
            torchmetrics.Accuracy(num_classes=opt.nclasses, average='macro').to(model.device),
            torchmetrics.IoU(num_classes=opt.nclasses).to(model.device),
            torchmetrics.F1(num_classes=opt.nclasses, average='macro').to(model.device)
        ]
        self.val_metrics = [
            torchmetrics.Accuracy(num_classes=opt.nclasses, average='macro').to(model.device),
            torchmetrics.IoU(num_classes=opt.nclasses).to(model.device),
            torchmetrics.F1(num_classes=opt.nclasses, average='macro').to(model.device)
        ]

    def training_step(self, batch, idx):
        self.model.set_input(batch)
        out = self.model.forward()
        loss = self.criterion(self.model.labels, out)

        pred_class = out.data.max(1)[1]
        not_padding = label_class != -1
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
        not_padding = label_class != -1
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
        opt = torch.optim.SGD(self.model.parameters(), lr=self.kwargs.get('learning_rate', 1e-3),
                              momentum=0.9,
                              weight_decay=0.0002)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.kwargs['max_epochs'] * 3)
        return [opt], [sched]


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--max_image_size', default=128)
    parser.add_argument('--num_classes', default=3)
    parser.add_argument('--pretrained', default=True)

    parser.add_argument('--train_data', default='../../data/windows/set_1/train')
    parser.add_argument('--test_data', default='../../data/windows/set_1/test')
    parser.add_argument('--label_file', default='../../data/windows/labels.txt')
    parser.add_argument('--train_augmentation', default=True)

    parser.add_argument('--progress_bar_refresh_rate', type=int, default=20)
    parser.add_argument('--default_root_dir', default='../../models/test_classification/densenet161/', help='pytorch-lightning log path')
    # parser.add_argument('--resume_from_checkpoint', default='../../models/test_classification/densenet121/lightning_logs/version_56/checkpoints/epoch=44-val_acc_epoch=0.98.ckpt')
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