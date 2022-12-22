import os.path as p
import numpy as np
import trimesh
from trimesh_wrapper import Trimesh_wrapper
from utils import *


def file_path(filename:str, sub_dir:str, file_type:str='') -> str:
    if sub_dir in ['inputs','i']:
        return f'{p.curdir}{p.sep}{"inputs"}{p.sep}{filename}.{file_type if file_type else "ply"}'
    elif sub_dir in ['outputs','o']:
        return f'{p.curdir}{p.sep}{"outputs"}{p.sep}{filename}.{file_type if file_type else "obj"}'
    else:
        return f'{p.curdir}{p.sep}{filename}'


def input_ply_path(ply_name:str) -> str:
    return file_path(ply_name,'i')


def output_obj_path(obj_name:str) -> str:
    return file_path(obj_name,'o')


def main(object_name: str) -> None:
    mesh = trimesh.load(input_ply_path(object_name))
    for facet in mesh.facets:
        mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    mesh.show()
    mesh.export(output_obj_path(object_name))


if __name__ == "__main__":
    main('bike-wheel')
    # main('copper-key')