import torch
import torch.nn as nn
from options.base_options import BaseOptions

class MeshUnpool(nn.Module):
    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target
        gpu_ids = BaseOptions().get_device()
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if len(gpu_ids)>0 else torch.device('cpu')

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group, unroll_start):
        start, end = group.shape
        padding_rows =  unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            size1 = group.shape[0] + padding_rows
            size2 = group.shape[1] + padding_cols
            group = torch.sparse_coo_tensor(
                indices=group._indices(), values=group._values(), dtype=torch.float32,
                size=(size1, size2))
        return group

    def pad_occurrences(self, occurrences):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    def forward(self, features, meshes):
        batch_size, nf, edges = features.shape
        groups = [self.pad_groups(mesh.get_groups(), edges).to(self.device) for mesh in meshes]
        unroll_mat = torch.stack(groups)
        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.unsqueeze(torch.stack(occurrences, dim=0), dim=1)
        imin = 0
        length = 500

        while imin <= unroll_mat.size()[2]:
            try:
                slicesgroup = unroll_mat.narrow_copy(2, imin, length).to_dense().to(self.device)
                occ = occurrences.narrow_copy(2, imin, length).to(self.device)
                res = (slicesgroup / occ).to_sparse()
                imin = imin + 500

                try:
                    allgroups = torch.cat((allgroups, res), -1)
                except NameError:
                    allgroups = res

            except Exception:
                length = unroll_mat.size()[2] - imin

        unroll_mat = allgroups.to(features.device)

        for mesh in meshes:
            mesh.unroll_gemm()
        one = unroll_mat.transpose(1,2)
        two = features.transpose(1,2)

        mauz =  torch.empty(size=(batch_size, features.shape[1], unroll_mat.shape[2]))

        for m in range(batch_size):
            ml = torch.matmul(one[m], two[m])
            if m == 0:
                mauz  = torch.unsqueeze(ml, dim=0)
            else:
                mauz = torch.cat((mauz, torch.unsqueeze(ml, dim=0)), dim=0)

        mauz = mauz.transpose(1,2)
        return mauz


