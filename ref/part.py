# encoding: utf-8
import os
import os.path as osp

import mmcv
import numpy as np

cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")

dataset_root = osp.join(bop_root, "part")
train_dir = osp.join(dataset_root, "train")
test_dir = osp.join(dataset_root, "test")

model_dir = osp.join(dataset_root, "models")
model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 1.0

objects = ["part"]
id2obj = {1: "part"}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]


def _try_get_diameters_from_models_info():
    models_info_path = osp.join(model_dir, "models_info.json")
    if not osp.exists(models_info_path):
        return np.array([0.1], dtype=np.float32)
    models_info = mmcv.load(models_info_path)
    diameters = []
    for obj_name in objects:
        obj_id = obj2id[obj_name]
        diameter_mm = float(models_info[str(obj_id)]["diameter"])
        diameters.append(diameter_mm / 1000.0)
    return np.array(diameters, dtype=np.float32)


diameters = _try_get_diameters_from_models_info()


def get_models_info():
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)
    return models_info


def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict

