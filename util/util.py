from __future__ import print_function
import torch
import numpy as np
import os
import torch.nn.functional as F

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

MESH_EXTENSIONS = [
    '.obj',
]

def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

def seg_accuracy(predicted, ssegs, meshes):
    correct = 0
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
        edge_areas = torch.from_numpy(mesh.get_edge_areas())
        correct += (correct_vec.float() * edge_areas).sum()
    return correct

def intersection_over_union(preds, target, num_classes,  batch=1):
        preds, target = F.one_hot(preds, num_classes), F.one_hot(target, num_classes)
        if batch is 1:
            i = (preds & target).sum(dim=0)
            u = (preds | target).sum(dim=0)
            iou = i.to(torch.float) / u.to(torch.float)
        else:
            iou = torch.zeros(num_classes, dtype=torch.float32)
            for idx, pred in enumerate(preds):
                i = (pred & target[idx]).sum(dim=0)
                u = (pred | target[idx]).sum(dim=0)
                iou = iou.add(i.to(torch.float) / u.to(torch.float))
        return iou

def mean_iou_calc(pred, target, num_classes, batch=None):
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    if (target == -1).any():
        target = target[target!=-1]
        len = target.size()
        pred = pred[:len[0]]

    iou = intersection_over_union(pred, target, num_classes, batch)
    print(iou)
    mean_iou = iou.mean(dim=-1)
    return mean_iou, iou

def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

#Select all rows by index
def myindexrowselect(groups, mask_index):

    index = groups._indices()
    newrowindex = -1

    print(groups, mask_index)

    for ind in mask_index:
        try:
            newrowindex = newrowindex + 1
        except NameError:
            newrowindex = 0

        keptindex = torch.squeeze((index[0] == ind).nonzero())

        if len(keptindex.size()) == 0:
            # Get column values from mask, create new row idx
            try:
                newidx = torch.cat((newidx, torch.tensor([newrowindex])), 0)
                newcolval = torch.cat((newcolval, torch.tensor([index[1][keptindex.item()]])), 0)
            except NameError:
                newidx = torch.tensor([newrowindex])
                newcolval = torch.tensor([index[1][keptindex.item()]])

        else:
            # Get column values from mask, create new row idx
            # Add newrowindex eee.size() time to list
            for i in range(list(keptindex.size())[0]):
                try:
                    newidx = torch.cat((newidx, torch.tensor([newrowindex])), 0)
                    newcolval = torch.cat((newcolval, torch.tensor([index[1][keptindex.tolist()[i]]])), 0)
                except NameError:
                    newidx = torch.tensor([newrowindex])
                    newcolval = torch.tensor([index[1][keptindex.tolist()[i]]])

    groups = torch.sparse_coo_tensor(indices=torch.stack((newidx, newcolval), dim=0),
                                     values=torch.ones(newidx.shape[0], dtype=torch.float),
                                     size=(len(mask_index), groups.shape[1]))
    return groups



def getsparserow(groups, ind):

#ind gibt index von row an
        index = groups._indices()
        keptindex = torch.squeeze((index[0] == ind).nonzero())


        if len(keptindex.size()) == 0:
            # Get column values from mask, create new row idx
            try:
                newidx = torch.tensor([0])
                newcolval = torch.cat((newcolval, torch.tensor([index[1][keptindex.item()]])), 0)
            except NameError:
                newcolval = torch.tensor([index[1][keptindex.item()]])
        else:
            # Get column values from mask, create new row idx
            # Add newrowindex eee.size() time to list
            for i in range(list(keptindex.size())[0]):
                try:
                    newidx = torch.cat((newidx,  torch.tensor([0])), 0)
                    newcolval = torch.cat((newcolval, torch.tensor([index[1][keptindex.tolist()[i]]])), 0)
                except NameError:
                    newidx = torch.tensor([0])
                    newcolval = torch.tensor([index[1][keptindex.tolist()[i]]])



        groups = torch.sparse_coo_tensor(indices=torch.stack((newidx, newcolval), dim=0),
                                     values=torch.ones(newcolval.shape[0], dtype=torch.float),
                                     size=(len(newcolval), groups.shape[1]))
        return groups