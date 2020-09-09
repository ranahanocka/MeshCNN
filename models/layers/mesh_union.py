import torch
from torch.nn import ConstantPad2d
import time
from util.util import  myindexrowselect

from options.base_options import BaseOptions

class MeshUnion:
    def __init__(self, n,  device=torch.device('cpu')):
        gpu_ids = BaseOptions().get_device()
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if len(gpu_ids)>0 else torch.device('cpu')

        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.values = torch.ones(n, dtype= torch.float)
        self.groups = torch.sparse_coo_tensor(indices= torch.stack((torch.arange(n), torch.arange(n)),dim=0), values= self.values,

                              size=(self.__size, self.__size), device=self.device)


    def union(self, source, target):
        index = torch.tensor([source], dtype=torch.long)
        row = myindexrowselect(self.groups, index, self.device).to(self.device)
        row._indices()[0] = torch.tensor(target)
        row = torch.sparse_coo_tensor(indices=row._indices(), values= row._values(),
                             size=(self.__size, self.__size), device=self.device)
        self.groups = self.groups.add(row)
        self.groups = self.groups.coalesce()
        del index, row


    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sparse.sum(self.groups, 0).values()


    def get_groups(self, tensor_mask):
        ## Max comp
        mask_index = torch.squeeze((tensor_mask == True).nonzero()).to(self.device)
        return myindexrowselect(self.groups, mask_index, self.device)


    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)

        self.groups = self.groups.to(self.device)
        fe = torch.matmul(self.groups.transpose(0,1),features.squeeze(-1).transpose(1,0)).transpose(0,1)
        occurrences = torch.sparse.sum(self.groups, 0).to_dense()
        fe = fe / occurrences
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe


    def prepare_groups(self, features, mask):
        mask_index = torch.squeeze((torch.from_numpy(mask) == True).nonzero())

        self.groups = myindexrowselect(self.groups, mask_index, self.device).transpose(1,0)
        padding_a = features.shape[1] - self.groups.shape[0]

        if padding_a > 0:
            self.groups = torch.sparse_coo_tensor(
                indices=self.groups._indices(),  values=self.groups._values(), dtype=torch.float32,
                size=(features.shape[1],  self.groups.shape[1]))


