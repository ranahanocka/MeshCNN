import numpy as np
import trimesh
from pykdtree.kdtree import KDTree


class MeshSDF:
    """
    This class is used to load a mesh from a file and sample points from its surface.
    It produces a point cloud of the surface and uses a KDTree to find the nearest neighbors of a point to estimate the
    SDF values for some sample points.
    :param point_cloud_path: path to the point cloud file
    :param num_samples: number of points to sample from the surface per batch_sample
    :param coarse_scale: ??
    :param fine_scale: ??
    :param num_closest_points: number of nearest neighbors to consider for SDF estimation
    """

    def __init__(
        self,
        point_cloud_path,
        num_samples=(30 ** 3) // 3,
        coarse_scale=1e-1,
        fine_scale=1e-3,
        num_closest_points=3,
    ):
        # params
        self.num_samples = num_samples
        self.num_closest_points = num_closest_points
        self.point_cloud_path = point_cloud_path
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale
        # data to be filled
        self.points = self.sdf = self.closest_points = None
        self.i = 0
        # finish initial setup
        self.load_mesh(point_cloud_path)
        self.sample_surface()

    def load_mesh(self, point_cloud_path=None):
        if not point_cloud_path:
            point_cloud_path = self.point_cloud_path
        if "xyz" in point_cloud_path:
            point_cloud = np.genfromtxt(point_cloud_path)
            self.v = point_cloud[:, :3]
            self.n = point_cloud[:, 3:]
        elif "obj" in point_cloud_path:
            point_cloud = trimesh.load(point_cloud_path)
            self.v = point_cloud.vertices
            self.n = point_cloud.vertex_normals
        else:
            raise NotImplementedError("Only xyz and obj files are supported")
        if self.v.shape[0] != self.n.shape[0]:
            raise ValueError(
                f"Point cloud vectors and normals have different number of points: {self.v.shape[0]} and {self.n.shape[0]}"
            )
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
        sdf, idx = self.kd_tree.query(points, k=self.num_closest_points)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]
        self.points = points
        self.sdf = sdf
        self.closest_points = self.v[idx]
        # shape closest points: (num_samples, num_closest_points, 3)
        self.i = 0

    def single_sample(self):
        if self.i >= self.num_samples:
            self.sample_surface()
        point, sdf = self.points[self.i], self.sdf[self.i]
        self.i += 1
        return point, sdf

    def single_sample_plus_nearest_neighbors(self):
        if self.i >= self.num_samples:
            self.sample_surface()
        point, sdf, nn = (
            self.points[self.i],
            self.sdf[self.i],
            self.closest_points[self.i, ...],
        )
        self.i += 1
        return point, sdf, nn
