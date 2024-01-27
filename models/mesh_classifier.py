from datetime import datetime
from os.path import join

import numpy as np
import torch
from torchmetrics.functional.classification import binary_confusion_matrix
from util.util import seg_accuracy, print_network
from . import networks


class ClassifierModel:
    """Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )
        self.save_dir = opt.expr_dir
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(
            opt.input_nc,
            opt.ncf,
            opt.ninput_edges,
            opt.nclasses,
            opt,
            self.gpu_ids,
            opt.arch,
            opt.init_type,
            opt.init_gain,
        )
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if opt.pretrained_path:
            self.load_network(opt.pretrained_path)
        elif not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data, inference=False):
        if type(data["edge_features"]) == torch.Tensor:
            input_edge_features = data["edge_features"].float()
        elif type(data["edge_features"]) == np.ndarray:
            input_edge_features = torch.from_numpy(data["edge_features"]).float()
        else:
            raise ValueError("edge_features must be either torch.Tensor or np.ndarray")
        self.edge_features = input_edge_features.to(self.device).requires_grad_(
            self.is_train
        )
        self.mesh = data["mesh"]
        if inference:
            return  # At inference time, we don't need to set labels.
        labels = None
        if self.opt.dataset_mode == "classification":
            labels = torch.from_numpy(data["label"]).long()
        elif self.opt.dataset_mode == "regression":
            labels = torch.from_numpy(data["regression_target"]).float()
        self.labels = labels.to(self.device)
        if self.opt.dataset_mode == "segmentation" and not self.is_train:
            self.soft_label = torch.from_numpy(data["soft_label"])

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    ##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = "%s_net.pth" % which_epoch
        if self.opt.pretrained_path:
            load_path = self.opt.pretrained_path
            self.opt.pretrained_path = None
        else:
            load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print("loading the model from %s" % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net.load_state_dict(state_dict, strict=False)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = "%s_net.pth" % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            if self.opt.dataset_mode == "classification":
                out = self.forward()
                # compute number of correct
                pred_class = out.data.max(1)[1]
                label_class = self.labels
                self.export_segmentation(pred_class.cpu())
                metrics = self.get_accuracy(pred_class, label_class)
            elif self.opt.dataset_mode == "regression":
                out = self.forward()
                # compute number of correct
                pred = out.data.cpu()
                label_class = self.labels.cpu()
                mae_times_n = self.get_accuracy(pred, label_class)
                sign_correct = self.get_accuracy(pred, label_class, "sign")
                pred_plus_minus_one = torch.where(
                    pred > 0, torch.ones_like(pred), torch.zeros_like(pred)
                )
                label_class_plus_minus_one = torch.where(
                    label_class > 0,
                    torch.ones_like(label_class),
                    torch.zeros_like(label_class),
                )
                conf_matrix = binary_confusion_matrix(
                    pred_plus_minus_one, label_class_plus_minus_one
                )
                metrics = dict(
                    mae_times_n=mae_times_n,
                    sign_correct=sign_correct,
                    conf_matrix=conf_matrix,
                )
        return metrics, len(label_class)

    def get_accuracy(self, pred, labels, mode=None):
        """computes accuracy for classification / segmentation"""
        mode = mode or self.opt.dataset_mode
        if mode == "classification":
            correct = pred.eq(labels).sum()
        elif mode == "segmentation":
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        elif mode == "regression":
            correct = abs(pred - labels).sum()
        elif mode == "sign":
            correct = pred.sign().eq(labels.sign()).sum()
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == "segmentation":
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
