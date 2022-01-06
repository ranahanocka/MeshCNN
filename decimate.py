from copy import deepcopy
import os
from collections import OrderedDict

import trimesh as tm
import numpy as np
import torch

from options.pl_options import PLOptions
from data import DataLoader
from data.segmentation_data import Mesh
from util.util import pad
from train_pl import MeshSegmenter


def show_mesh(mesh, label):
    edges = mesh.edges
    vertices = mesh.vs
    vertex_label = np.zeros(len(vertices))
    for e_l, e in zip(label[0], edges):
        if e_l == 1:
            vertex_label[e] = 1
    faces = mesh.faces
    vertex_colors = np.array([[255, 100, 0, 255], [0, 100, 255, 255]])[vertex_label.astype(int)]
    trimesh = tm.Trimesh(faces=faces, vertices=vertices, vertex_colors=vertex_colors)
    trimesh.show()
    return trimesh


def simplify_rooftop(roof_segment: tm.Trimesh, n_triangles) -> tm.Trimesh:
    """
    Perform mesh simplificaiton based on desired triangles number
    :param roof_segment: Trimesh - submesh of the roof
    :param n_triangles: int - number of triangles the simplified mesh would contain
    :return: tm.Trimesh - Simplified mesh
    """
    n_triangles = max([n_triangles, 5])
    segment = roof_segment.simplify_quadratic_decimation(n_triangles)

    return segment


def load_obj(path, opt, mean, std):
    mesh = Mesh(file=path, opt=opt, hold_history=True, export_folder=opt.export_folder)
    meta = {}
    meta['mesh'] = [mesh]
    meta['path'] = [path]
    edge_features = mesh.extract_features()
    edge_features = pad(edge_features, opt.ninput_edges)
    edge_features = (edge_features - mean) / std
    meta['edge_features'] = np.expand_dims(edge_features, 0)
    meta['label'] = np.array([])
    meta['soft_label'] = np.array([])
    return meta


def run_decimation(epoch=-1):
    opt = PLOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = MeshSegmenter(opt)

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    checkpoint = torch.load(opt.model_path, map_location=device)

    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('.module', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    for i, data in enumerate(dataset):
        if i != 21 and i != 22:
            continue
        print(i, data['path'])
        obj_name = os.path.basename(data['path'][0])
        print(obj_name)

        torch.cuda.empty_cache()

        mesh = deepcopy(data['mesh'][0])
        pred_class = model.forward(data).max(1)[1]
        tm_mesh = show_mesh(mesh, label=pred_class)

        torch.cuda.empty_cache()

        for desired_triangle_area in [0.3, 8]:#[0.5, 0.8, 1, 1.2, 1.4, 1.7, 2, 2.5, 3.5, 5, 7]:
            print(desired_triangle_area)

            tm_mesh_new = simplify_rooftop(tm_mesh, int((tm_mesh.area / desired_triangle_area)))

            new_obj_name = obj_name[:-4] + '_' + str(desired_triangle_area) + '.obj'
            obj_path = os.path.join(opt.decimation_dir, new_obj_name)

            with open(obj_path, mode='w') as f:
                f.write(tm.exchange.obj.export_obj(tm_mesh_new))
                data_new = load_obj(obj_path, opt, dataset.dataset.mean, dataset.dataset.std)

                mesh_new = deepcopy(data_new['mesh'][0])
                pred_class_new = model.forward(data_new).max(1)[1]
                show_mesh(mesh_new, label=pred_class_new)

                torch.cuda.empty_cache()
            # os.unlink(f.name)

if __name__ == '__main__':
    run_decimation()
