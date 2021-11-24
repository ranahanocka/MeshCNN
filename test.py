from copy import deepcopy

import numpy as np
import torch
import torchmetrics
import trimesh as tm

from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


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


# def run_test(epoch=-1):
#     print('Running Test')
#     opt = TestOptions().parse()
#     opt.serial_batches = True  # no shuffle
#     dataset = DataLoader(opt)
#     model = create_model(opt)
#     writer = Writer(opt)
#     # test
#     writer.reset_counter()
#     for i, data in enumerate(dataset):
#         mesh = deepcopy(data['mesh'][0])
#         model.set_input(data)
#
#         # pred_class = model.forward().max(1)[1]
#         # # show_mesh(mesh, pred_class[0])
#         # edges = mesh.edges
#         # vertices = mesh.vs
#         # vertex_label = np.zeros(len(vertices))
#         # for e_l, e in zip(pred_class[0], edges):
#         #     if e_l == 1:
#         #         vertex_label[e] = 1
#         # faces = mesh.faces
#         # vertex_colors = np.array([tm.visual.random_color(), tm.visual.random_color()])[vertex_label.astype(int)]
#         # tm.Trimesh(faces=faces, vertices=vertices, vertex_colors=vertex_colors).show()
#
#         ncorrect, nexamples = model.test()
#
#         writer.update_counter(ncorrect, nexamples)
#     writer.print_acc(epoch, writer.acc)
#     return writer.acc


def run_test(epoch=-1, data_phase='test'):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.phase = data_phase
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    acc_metric = torchmetrics.Accuracy(num_classes=2, average='macro').to(model.device)
    iou_metric = torchmetrics.IoU(num_classes=2).to(model.device)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            out = model.forward()
            pred_class = out.data.max(1)[1]
            label_class = model.labels
            pred_class = pred_class.to(model.device)
            label_class = label_class.to(model.device)
            not_padding = label_class != -1
            label_class = label_class[not_padding]
            pred_class = pred_class[not_padding]

            acc = acc_metric(pred_class, label_class)
            iou = iou_metric(pred_class, label_class)
            # print(f"Metrics on 3D model {i} - accuracy: {acc}, F1: {f1}, IoU: {iou}")
    # writer.print_acc(epoch, writer.acc)
    total_acc = acc_metric.compute()
    total_iou = iou_metric.compute()
    print(f'epoch: {epoch}, {data_phase.upper()} ACC: {total_acc}, IoU: {total_iou}')


if __name__ == '__main__':
    run_test()