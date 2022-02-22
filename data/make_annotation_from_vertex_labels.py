import numpy as np
import os
import glob
import filecmp
import sys


'''
Creates esseg files for accuracy with smooth transitions between classes 
Requires Objects and corresponding labels per edge
Author: Rana Hanocka / Lisa Schneider

@input: 
    <input_path> path where seg, sseg, train, test folders are placed 

@output:
    esseg files for all objects
    to run it from cmd line:
    python create_sseg.py /home/user/MedMeshCNN/datasets/human_seg/
'''

def compute_face_normals_and_areas(vs, faces):
    face_normals = np.cross(vs[faces[:, 1]] - vs[faces[:, 0]],
                            vs[faces[:, 2]] - vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face'
    face_areas *= 0.5
    return face_normals, face_areas


def remove_non_manifolds(vs, faces):
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(vs, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]

def get_gemm_edges(faces, export_name_edges):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                nb_count.append(0)
                edges_count += 1
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    edges = np.array(edges, dtype=np.int32)
    np.savetxt(export_name_edges, edges, fmt='%i')
    return edge_nb, edges


def load_labels(path):
    with open(path, 'r') as f:
        content = f.read().splitlines()
        return content

def create_sseg_file(gemms, labels, export_name_seseg):
    gemmlabels = {}
    classes = len(np.unique(labels))
    class_to_idx = {v: i for i, v in enumerate(np.unique(labels))}
    totaledges = len(gemms)
    sseg = np.zeros([ totaledges, classes])
    for i, edges in enumerate(gemms):
        alllabels = []
        for edge in range(len(edges)):
            lookupEdge = edges[edge]
            label = labels[lookupEdge]
            alllabels.append(label)
        gemmlabels[i] = alllabels

    for i, edges in enumerate(gemms):
            gemmlab = gemmlabels[i]
            uniqueValues, counts = np.unique(gemmlab, return_counts=True)
            for j, label in enumerate(uniqueValues):
                weight = 0.125*counts[j]
                sseg[i][class_to_idx[label]] = weight
    np.savetxt(export_name_seseg, sseg,  fmt='%1.6f')

def get_obj(file):
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return faces, vs


import trimesh as tm
def edges_to_path(edges, color=tm.visual.color.random_color()):
    lines = np.asarray(edges)
    args = tm.path.exchange.misc.lines_to_path(lines)
    colors = [color for _ in range(len(args['entities']))]
    path = tm.path.Path3D(**args, colors=colors)
    return path


def show_mesh(edges, vs, label,  colors=[[0,0,0,255], [120,120,120,255]]):
    colors = np.array(colors)
    edges = vs[edges]
    tm.Scene([edges_to_path(e, colors[int(l)]) for e, l in zip(edges, label)]).show()



def create_files(path):
    for filename in glob.glob(os.path.join(path, 'obj/*.obj')):
        print(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        v_label_name = os.path.join(os.path.join(path, 'vseg'), basename + '.eseg')
        label_name = os.path.join(os.path.join(path, 'seg'), basename + '.eseg')
        export_name_seseg = os.path.join(os.path.join(path, 'sseg'), basename + '.seseg')
        export_name_edges = os.path.join(os.path.join(path, 'edges'), basename + '.edges')

        faces, vs = get_obj(filename)
        faces, face_areas = remove_non_manifolds(vs, faces)
        gemms, edges = get_gemm_edges(faces, export_name_edges)
        with open(v_label_name) as f:
            v_label = np.array(f.readlines(), dtype=int)

        edge_label = []
        for e in edges:
            if v_label[e[0]] == 1 and v_label[e[1]] == 1:
                edge_label.append(str(2))
            else:
                edge_label.append(str(1))

        with open(label_name, 'w') as f:
            f.write('\n'.join(edge_label))
        print(len(edge_label))
        if os.path.isfile(label_name):

            create_sseg_file(gemms, edge_label, export_name_seseg)
        else:
            print(label_name, "is no directory")


if __name__ == '__main__':
    create_files(sys.argv[1])