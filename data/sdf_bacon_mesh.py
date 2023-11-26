import os

import numpy as np
import torch
import torch.utils.data as data
import trimesh
from pykdtree.kdtree import KDTree

from data.base_dataset import BaseDataset
from data.positional_encoding import PositionalEncoding3D


class SdfDataset(BaseDataset):
    """
    Deprecated use sdf_regression_data file
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.get_mean_std()
        self.output_dimensions = opt.output_dimension
        self.paths = []
        for x in os.walk(self.dir):
            dir, direcs, paths = x
            for path in paths:
                if path.endswith(".xyz"):
                    self.paths.append(os.path.join(dir, path))

        self.dataset_length = len(self.paths)
        self.num_samples = opt.num_samples
        self.meshes = [MeshSDF(os.path.join(self.dir, path)) for path in self.paths]
        [mesh.load_mesh() for mesh in self.meshes]
        print(
            "Initialized SDF Dataset and loaded point clouds from {}".format(self.dir)
        )
        max_freq_log2 = 5
        num_freqs = 10
        log_sampling = True
        opt.max_freq_log2 = max_freq_log2
        opt.num_freqs = num_freqs
        opt.log_sampling = log_sampling

        self.positional_encoder = PositionalEncoding3D(opt)

    def __len__(self):
        return self.dataset_length

    """ Gets a sample for training sdf 
    """

    def __getitem__(self, index):
        point, sdf = next(self.meshes[index % self.dataset_length])
        point = torch.from_numpy(point).float()
        # point: (3) to (1, 3)
        point = point.unsqueeze(0)
        point_encoding = self.positional_encoder.forward(point)[0]
        # TODO try to do it without encoding only xyz coordinates


class MeshSDF(data.Dataset):
    """convert point cloud to SDF"""

    def __init__(
        self,
        pointcloud_path,
        num_samples=(30 ** 3) // 3,
        coarse_scale=1e-1,
        fine_scale=1e-3,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.pointcloud_path = pointcloud_path
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale
        self.points = None
        self.sdf = None
        self.i = 0
        self.load_mesh(pointcloud_path)
        self.sample_surface()

    def __len__(self):
        return self.num_samples  # arbitrary

    def load_mesh(self, pointcloud_path=None):
        if not pointcloud_path:
            pointcloud_path = self.pointcloud_path
        if "xyz" in pointcloud_path:
            pointcloud = np.genfromtxt(pointcloud_path)
            self.v = pointcloud[:, :3]
            self.n = pointcloud[:, 3:]
        elif "obj" in pointcloud_path:
            pointcloud = trimesh.load(pointcloud_path)
            self.v = pointcloud.vertices
            self.n = pointcloud.vertex_normals
        else:
            raise NotImplementedError("Only xyz and obj files are supported")

        n_norm = np.linalg.norm(self.n, axis=-1)[:, None]
        n_norm[n_norm == 0] = 1.0
        self.n = self.n / n_norm
        self.v = self.normalize(self.v)
        self.kd_tree = KDTree(self.v)

    def normalize(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def sample_surface(self):
        idx = np.random.randint(0, self.v.shape[0], self.num_samples)
        points = self.v[idx]
        points[::2] += np.random.laplace(
            scale=self.coarse_scale, size=(points.shape[0] // 2, points.shape[-1])
        )
        points[1::2] += np.random.laplace(
            scale=self.fine_scale, size=(points.shape[0] // 2, points.shape[-1])
        )

        # wrap around any points that are sampled out of bounds
        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.kd_tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]
        self.points = points
        self.sdf = sdf
        self.i = 0

    def single_sample(self):
        if self.i >= self.num_samples:
            self.sample_surface()
        point, sdf = self.points[self.i], self.sdf[self.i]
        self.i += 1
        return point, sdf
