import json
import os


def convert_xyz_to_obj(xyz_file_path, obj_file_path):
    with open(xyz_file_path, "r") as xyz_file, open(obj_file_path, "w") as obj_file:
        obj_file.write("# OBJ file\n")
        for line in xyz_file:
            x, y, z, _, _, _ = line.split()
            obj_file.write(f"v {x} {y} {z}\n")


def convert_json_to_obj(json_file_path, obj_file_path):
    if os.path.exists(obj_file_path):
        raise FileExistsError(f"{obj_file_path} already exists")
    with open(json_file_path, "r") as json_file, open(obj_file_path, "w+") as obj_file:
        obj_file.write("# OBJ file\n")
        json_line = json_file.readline()
        parsed_json = json.loads(json_line)
        strin = ""
        v = 0
        for vertex in parsed_json[1]:
            if isinstance(vertex, str):
                continue
            x = vertex[1]
            y = vertex[2]
            z = vertex[3]
            strin += f"v {x} {y} {z}\n"
            v += 1

        strin += f"# {v} vertices, 0 vertices normals\n\n"
        f = 0
        for face in parsed_json[2][1][1]:
            if isinstance(face, str):
                continue
            v1 = face[1]
            v2 = face[2]
            v3 = face[3]

            strin += f"f {v1} {v2} {v3}\n"
            f += 1
        strin += f"# {f} faces, 0 coords texture\n"
        obj_file.write(
            f"""####
#
# OBJ File Generated by Convert Script in MeshCNN Repo
#
####
# Object {os.path.basename(json_file_path)}.obj
#
# Vertices: {v}
# Faces: {f}
#
####
"""
        )
        obj_file.write(strin)
        obj_file.write(
            """
# End of File"""
        )


convert_json_to_obj(
    "/Users/julianstrietzel/PycharmProjects/MeshCNN/datasets/armadillo_only_for_overfitting/Stanford-Armadillo.json",
    "/Users/julianstrietzel/PycharmProjects/MeshCNN/datasets/armadillo_only_for_overfitting/gt_armadillo.json.obj",
)