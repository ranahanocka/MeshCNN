import os
import time
import torch
import numpy as np

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.iou = torch.zeros([opt.nclasses]).to(self.device)
        self.avg_iou = 0

        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """ prints train loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                  % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)

    def plot_lr(self, lr, epoch):
        if self.display:
            self.display.add_scalar('data/lr', lr, epoch)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST ACC: [{:.5} %]' \
            .format(epoch, acc * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_iou(self, epoch, avg_iou, iou):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST MEAN_IOU:  [{:.5} %] \nepoch: {}, IOU: {}\n' \
            .format(epoch, avg_iou * 100, epoch, iou.cpu().numpy() * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s' % message)

    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc[0], epoch)

    def reset_counter(self, opt):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0
        self.iou = torch.zeros([opt.nclasses])
        self.avg_iou = 0

    def update_counter(self, ncorrect, nexamples, avg_iou, iou):
        self.ncorrect += ncorrect
        self.nexamples += nexamples
        self.iou = torch.stack([self.iou.to(self.device), iou.to(self.device)]).sum(dim=0)
        self.avg_iou += avg_iou

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    @property
    def mean_iou(self):
        return (self.avg_iou / self.nexamples)

    @property
    def seg_iou(self):
        return (self.iou / self.nexamples)

    def close(self):
        if self.display is not None:
            self.display.close()
