from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import trimesh as tm
import numpy as np
from copy import deepcopy


import trimesh as tm
def edges_to_path(edges, color=tm.visual.color.random_color()):
    lines = np.asarray(edges)
    args = tm.path.exchange.misc.lines_to_path(lines)
    colors = [color for _ in range(len(args['entities']))]
    path = tm.path.Path3D(**args, colors=colors)
    return path


def show_edges(mesh, label, colors=[[0,0,0,255], [120,120,120,255]]):
    colors = np.array(colors)
    edges = mesh.vs[mesh.edges]
    tm.Scene([edges_to_path(e, colors[int(l)]) for e, l in zip(edges, label)]).show()




def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        mesh = deepcopy(data['mesh'][0])
        model.set_input(data)
        pred_class = model.forward().max(1)[1]
        # show_mesh(mesh, pred_class[0])
        edges = mesh.edges
        vertices = mesh.vs
        vertex_label = np.zeros(len(vertices))
        for e_l, e in zip(pred_class[0], edges):
            if e_l == 1:
                vertex_label[e] = 1
        faces = mesh.faces
        vertex_colors = np.array([tm.visual.random_color(), tm.visual.random_color()])[vertex_label.astype(int)]
        tm.Trimesh(faces=faces, vertices=vertices, vertex_colors=vertex_colors).show()
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
