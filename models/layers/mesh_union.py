import torch
from torch.nn import ConstantPad2d
from util.util import myindexrowselect



class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.values = torch.ones(n, dtype= torch.float)
        self.groups = torch.sparse_coo_tensor(indices= torch.stack((torch.arange(n), torch.arange(n)),dim=0), values= self.values,
                              size=(self.__size, self.__size), device=device)



    def union(self, source, target):
        #Get source row
        index = torch.tensor([source], dtype=torch.long)
        row = myindexrowselect(self.groups, index)

        #Change to target row
        row._indices()[0] = torch.tensor(target)
        row = torch.sparse_coo_tensor(indices=row._indices(), values= row._values(),
                             size=(self.__size, self.__size))

        #Add to target row
        self.groups = self.groups.add(row)
        self.groups = self.groups.coalesce()

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sparse.sum(self.groups, 0).values()


    def get_groups(self, tensor_mask):
        ## Max comp
        mask_index = torch.squeeze((tensor_mask == True).nonzero())
        groups = myindexrowselect(self.groups,mask_index)
        return groups


    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)

        features = features.type(torch.FloatTensor)
        fe = torch.matmul(self.groups.transpose(0,1),features.squeeze(-1).transpose(1,0)).transpose(0,1)
        occurrences = torch.sparse.sum(self.groups, 0).to_dense()
        fe = fe / occurrences
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe


    def prepare_groups(self, features, mask):
        tensor_mask = torch.from_numpy(mask)
        mask_index = torch.squeeze((tensor_mask == True).nonzero())

        self.groups = myindexrowselect(self.groups, mask_index)
        self.groups = self.groups.transpose(1,0)

        padding_a = features.shape[1] - self.groups.shape[0]

        if padding_a > 0:
            self.groups = torch.sparse_coo_tensor(
                indices=self.groups._indices(),  values=self.groups._values(), dtype=torch.float32,
                size=(features.shape[1],  self.groups.shape[1]))


