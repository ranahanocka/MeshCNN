from __future__ import print_function
import torch
import numpy as np
import os


# from torch_scatter import scatter_add

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


def intersection_over_union(preds, target, num_classes):
    preds, target = torch.nn.functional.one_hot(preds, num_classes), torch.nn.functional.one_hot(target, num_classes)
    iou = torch.zeros(num_classes, dtype=torch.float32)
    for idx, pred in enumerate(preds):
            i = (pred & target[idx]).sum(dim=0)
            u = (pred | target[idx]).sum(dim=0)
            iou = iou.add(i.cpu().to(torch.float) / u.cpu().to(torch.float))
    return iou


def mean_iou_calc(pred, target, num_classes):
    #Removal of padded labels marked with -1
    slimpred = []
    slimtarget = []

    for batch in range(pred.shape[0]):
        if (target[batch] == -1).any():
            slimLabels = target[batch][target[batch]!=-1]
            slimtarget.append(slimLabels)
            slimpred.append(pred[batch][:slimLabels.size()[0]])

    pred = torch.stack(slimpred,0)
    target = torch.stack(slimtarget, 0)

    iou = intersection_over_union(pred, target, num_classes)
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
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
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





def myindexrowselect(groups, mask_index, device):

    sparseIndices = groups._indices()
    newIndices = []

    for i, value in enumerate(mask_index):
        #Get index from relevant indices
        index = (sparseIndices[0] == value).nonzero()

        #Get rows by index
        sparseRow = [sparseIndices[:, value] for value in index]
        sparseRow = torch.cat(sparseRow,1)[1]
        singleRowIndices = torch.squeeze(torch.full((1,len(sparseRow)),i, dtype=torch.long),0).to(sparseRow.device)
        indices = torch.stack((singleRowIndices,sparseRow))
        newIndices.append(indices)

        allNewIndices = torch.cat(newIndices,1)

    #Create new tensor
    groups = torch.sparse_coo_tensor(indices=allNewIndices,
                                     values=torch.ones(allNewIndices.shape[1], dtype=torch.float),
                                     size=(len(mask_index), groups.shape[1]))

    return groups
