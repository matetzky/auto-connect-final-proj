import os.path as p
import numpy as np
from mesh_utils import *
import trimesh
from trimesh_handler import Trimesh_handler


def file_path_handler(filename, sub_dir, file_type=''):
    if sub_dir in ['plys','i']:
        return f'{p.curdir}{p.sep}{"plys"}{p.sep}{filename}.{file_type if file_type else "ply"}'
    elif sub_dir in ['outputs','o']:
        return f'{p.curdir}{p.sep}{"outputs"}{p.sep}{filename}.{file_type if file_type else "obj"}'
    else:
        return f'{p.curdir}{p.sep}{filename}'


def input_ply_path(ply_name):
    return file_path_handler(ply_name,'i')


def output_obj_path(obj_name):
    return file_path_handler(obj_name,'o')


def main(object_name):
    mesh = trimesh.load(input_ply_path(object_name))
    for facet in mesh.facets:
        mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    mesh.show()
    mesh.export(output_obj_path(object_name))


if __name__ == "__main__":
    main('apple')
