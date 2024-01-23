import os

import numpy as np
import torch

from data.base_dataset import BaseDataset
from data.positional_encoding import point_encoder_fabric
from data.sdf_bacon_mesh import MeshSDF
from models.layers.mesh import Mesh
from util.util import is_mesh_file, pad, MESH_EXTENSIONS


class RegressionDataset(BaseDataset):
    def __init__(self, opt, path=None):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.root = opt.dataroot
        if path:
            self.paths = [(path, 0)]
        else:
            dire = os.path.join(opt.dataroot)
            if not os.path.exists(dire):
                raise FileNotFoundError(
                    f"Could not find data directory {opt.dataroot} or {dire}"
                )
            _, class_to_idx = self.find_classes(dire)
            self.paths = self.make_dataset_by_class(dire, class_to_idx, opt.phase)
        self.nclasses = 1
        self.size = len(self.paths)
        self.sdf_meshes = [MeshSDF(path) for path, _ in self.paths]
        # modify for network later.
        opt.nclasses = self.nclasses
        self.positional_encoder = point_encoder_fabric(opt)

        opt.input_nc = self.ninput_channels = 5 + 3 * opt.encoding_factor
        self.meshes = [
            Mesh(
                file=path,
                opt=opt,
                hold_history=False,
                export_folder=self.opt.export_folder,
            )
            for path, _ in self.paths
        ]
        self.mean_defined = False
        self.get_mean_std()

    def __getitem__(self, index):
        index = index % self.size
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = self.meshes[index]
        # get edge features
        normed_edge_features = self.get_normed_edge_features(mesh)
        if not self.mean_defined:
            meta = {"mesh": mesh, "label": label, "edge_features": normed_edge_features}
            return meta

        # get sdf mesh and sample a point for regression target and positional encoding
        sdf_mesh = self.sdf_meshes[index % self.size]
        point, sdf = sdf_mesh.single_sample()
        positional_encoded_point = self.positional_encoder.forward(
            torch.from_numpy(np.expand_dims(point, 0))
        ).float()

        positional_encoded_point = (
            positional_encoded_point[0]
            if len(positional_encoded_point.shape) == 2
            else positional_encoded_point
        )
        # positional_encoded_point has shape (3) should be (3, 750) to concat with edge features
        positional_encoded_point_repeated = np.repeat(
            np.expand_dims(positional_encoded_point, 1),
            normed_edge_features.shape[1],
            axis=1,
        )

        meta = {
            "mesh": mesh,
            "label": label,
            "regression_target": sdf,
            "edge_features": np.concatenate(
                (normed_edge_features, positional_encoded_point_repeated), axis=0,
            ),
        }
        return meta

    def get_normed_edge_features(self, mesh: Mesh):
        if mesh.normed is not None:
            return mesh.normed
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        if not self.mean_defined:
            return edge_features
        normed_edge_features = (edge_features - self.mean) / self.std
        mesh.normed = normed_edge_features
        return normed_edge_features

    def __len__(self):
        return self.size * self.opt.batch_size * 100

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, 1)
                    meshes.append(item)
        if len(meshes) == 0:
            raise RuntimeError(
                "Found 0 meshes in subfolders of: " + dir + "\n"
                "Supported mesh extensions are: " + ",".join(MESH_EXTENSIONS)
            )
        return meshes
