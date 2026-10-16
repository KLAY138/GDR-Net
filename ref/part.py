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
vertex_scale = 0.001

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
        return np.array([0.1], dtype=np.float64)
    models_info = mmcv.load(models_info_path)
    diameters = []
    for obj_name in objects:
        obj_id = obj2id[obj_name]
        diameter_mm = float(models_info[str(obj_id)]["diameter"])
        diameters.append(diameter_mm / 1000.0)
    return np.array(diameters, dtype=np.float64)


diameters = _try_get_diameters_from_models_info()

def _scale_models_info(models_info, scale):
    scaled = {}
    for obj_id, info in models_info.items():
        if not isinstance(info, dict):
            scaled[obj_id] = info
            continue
        new_info = dict(info)
        for k in ["diameter", "size_x", "size_y", "size_z", "min_x", "min_y", "min_z"]:
            if k in new_info:
                new_info[k] = float(new_info[k]) * scale
        if "symmetries_discrete" in new_info:
            new_syms = []
            for sym in new_info["symmetries_discrete"]:
                sym_4x4 = np.array(sym, dtype=np.float64).reshape(4, 4)
                sym_4x4[:3, 3] *= scale
                new_syms.append(sym_4x4.reshape(-1).tolist())
            new_info["symmetries_discrete"] = new_syms
        if "symmetries_continuous" in new_info:
            new_syms = []
            for sym in new_info["symmetries_continuous"]:
                new_sym = dict(sym)
                if "offset" in new_sym:
                    new_sym["offset"] = (np.array(new_sym["offset"], dtype=np.float64) * scale).tolist()
                new_syms.append(new_sym)
            new_info["symmetries_continuous"] = new_syms
        scaled[obj_id] = new_info
    return scaled


def get_models_info():
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)
    return _scale_models_info(models_info, vertex_scale)


def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict

