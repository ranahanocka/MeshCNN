import os

import mcubes
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from data import DataLoader
from data.sdf_regression_data import RegressionDataset
from models import create_model
from models.layers.mesh import Mesh
from options.test_options import TestOptions
from util.writer import Writer


def run_test(epoch=-1):
    print("Running Test")
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        mae, nexamples = model.test()
        writer.update_counter(mae, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


def rebuild_shape(grid_size=100, opt=None, part_to_sample: float = 0.75):
    opt = opt or TestOptions().parse()
    # first object in dataroot path
    xyz_path = os.path.join(
        opt.dataroot,
        list(filter(lambda x: str(x).endswith(".xyz"), os.listdir(opt.dataroot)))[0],
    )
    dataset = RegressionDataset(opt, path=xyz_path)
    pos_encoder = dataset.positional_encoder
    obj_path = xyz_path.replace(".xyz", ".obj")
    mesh = Mesh(file=obj_path, opt=opt, hold_history=False, export_folder=None,)
    normed_edge_features = dataset.get_normed_edge_features(mesh)
    model = create_model(opt)

    batch_size = 256
    batched_mesh = np.array([mesh] * batch_size)
    normed_edge_features_batched = np.repeat(
        np.expand_dims(normed_edge_features, 0), batch_size, axis=0
    )
    normed_edge_features_batched = torch.from_numpy(
        normed_edge_features_batched
    ).float()

    def mesh_cnn_sampling(points):
        pos_encoded_points = pos_encoder.forward(torch.from_numpy(points)).float()
        expanded_pos_encoded_points = torch.unsqueeze(pos_encoded_points, 2)

        # Adjust batch size if number of points is less than batch size
        current_batch_size = min(batch_size, points.shape[0])

        positional_encoded_point_repeated = expanded_pos_encoded_points.repeat(
            1, 1, normed_edge_features.shape[1]
        )
        all_edge_features = torch.cat(
            (
                normed_edge_features_batched[:current_batch_size],
                positional_encoded_point_repeated,
            ),
            dim=1,
        )
        batched_meta = {
            "mesh": batched_mesh[:current_batch_size],
            "edge_features": all_edge_features,
        }
        model.set_input(batched_meta, inference=True)
        with torch.no_grad():
            sdf = model.forward().data.cpu().numpy()
        return sdf

    # Step 1: Generate a 3D grid of points
    x, y, z = np.meshgrid(
        np.linspace(-part_to_sample, part_to_sample, grid_size),
        np.linspace(-part_to_sample, part_to_sample, grid_size),
        np.linspace(-part_to_sample, part_to_sample, grid_size),
    )

    # Flatten the grid to pass to the sampling function
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Step 2: Pass the grid of points to the mesh_cnn_sampling function
    # Before batch points to batch size
    sdf_values = None
    for i in tqdm(range(0, len(points), batch_size)):
        sdf_values_new = mesh_cnn_sampling(points[i : i + batch_size])
        if i == 0:
            sdf_values = sdf_values_new
        else:
            sdf_values = np.concatenate((sdf_values, sdf_values_new), axis=0)

    # Step 3: Reshape the SDF values into a 3D array
    sdf_values_3d = sdf_values.reshape((grid_size, grid_size, grid_size))

    # Step 4: Pass the 3D array of SDF values to the Marching Cubes algorithm
    vertices, triangles = mcubes.marching_cubes(-sdf_values_3d, 0)

    # Create a trimesh object and adjust the vertices
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # Export the mesh
    os.makedirs("./outputs/meshes", exist_ok=True)
    mesh.export(f"./outputs/meshes/{opt.name}_{grid_size}_{opt.timestamp}.obj")


if __name__ == "__main__":
    opt = TestOptions().parse()
    for grid_size in [25, 50, 100, 200]:
        print(f"Rebuilding shape for grid size {grid_size}")
        rebuild_shape(grid_size=grid_size, opt=opt)
