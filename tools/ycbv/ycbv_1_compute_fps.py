import os.path as osp
import sys
import pickle
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))

try:
    import mmcv  # type: ignore
except Exception:
    mmcv = None

try:
    from lib.pysixd import inout  # type: ignore
except Exception:
    inout = None

try:
    import ref  # type: ignore
except Exception:
    ref = None


def _load_ply_xyz_binary_little_endian(path: str) -> "np.ndarray":
    import numpy as np

    header = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("invalid ply: missing end_header")
            s = line.decode("ascii", "ignore").strip()
            header.append(s)
            if s == "end_header":
                break
        fmt = [l for l in header if l.startswith("format")][0].split()[1]
        if fmt != "binary_little_endian":
            raise RuntimeError(f"only binary_little_endian ply is supported, got: {fmt}")

        vcount = int([l for l in header if l.startswith("element vertex")][0].split()[2])
        props = []
        in_vertex = False
        for l in header:
            if l.startswith("element vertex"):
                in_vertex = True
                continue
            if l.startswith("element") and not l.startswith("element vertex"):
                in_vertex = False
            if in_vertex and l.startswith("property"):
                parts = l.split()
                if parts[1] == "list":
                    continue
                props.append((parts[2], parts[1]))

        type_map = {
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
            "uchar": "u1",
            "uint8": "u1",
            "char": "i1",
            "int8": "i1",
            "ushort": "<u2",
            "uint16": "<u2",
            "short": "<i2",
            "int16": "<i2",
            "uint": "<u4",
            "uint32": "<u4",
            "int": "<i4",
            "int32": "<i4",
        }
        dtype = np.dtype([(name, type_map[typ]) for name, typ in props])
        data = np.fromfile(f, dtype=dtype, count=vcount)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    return xyz


def _farthest_point_sampling_np(pts: "np.ndarray", num_fps: int, init_center: bool) -> "np.ndarray":
    import numpy as np

    if pts.shape[0] == 0:
        raise RuntimeError("empty point set")

    if pts.shape[0] > 20000:
        idx = np.random.choice(pts.shape[0], size=20000, replace=False)
        pts = pts[idx]

    if init_center:
        first = pts.mean(axis=0, keepdims=True)
        selected = [first[0]]
        min_d2 = np.sum((pts - first) ** 2, axis=1)
    else:
        i0 = int(np.random.randint(0, pts.shape[0]))
        selected = [pts[i0]]
        min_d2 = np.sum((pts - pts[i0]) ** 2, axis=1)

    for _ in range(num_fps):
        i = int(np.argmax(min_d2))
        p = pts[i]
        selected.append(p)
        d2 = np.sum((pts - p) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    return np.stack(selected[1:], axis=0).astype(np.float64)


def _get_fps_and_center_fallback(pts: "np.ndarray", num_fps: int, init_center: bool) -> "np.ndarray":
    import numpy as np

    fps_pts = _farthest_point_sampling_np(pts, num_fps=num_fps, init_center=init_center)
    center = pts.mean(axis=0, keepdims=True).astype(np.float64)
    return np.concatenate([fps_pts, center], axis=0)


def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-key", default="ycbv", type=str)
    parser.add_argument("--vertex-scale", default=0.001, type=float)
    parser.add_argument("--model-dir", default="", type=str)
    parser.add_argument("--obj-ids", default="", type=str)
    args = parser.parse_args()

    if ref is not None:
        data_ref = ref.__dict__[args.ref_key]
        model_dir = data_ref.model_dir
        obj_ids = sorted(list(data_ref.id2obj.keys()))
    else:
        if not args.model_dir:
            raise RuntimeError("mmcv/ref not available; please provide --model-dir")
        model_dir = args.model_dir
        if args.obj_ids:
            obj_ids = [int(x) for x in args.obj_ids.split(",") if x.strip()]
        else:
            obj_ids = [1]

    vertex_scale = args.vertex_scale
    fps_dict = {}
    for obj_id in tqdm(obj_ids):
        print(obj_id)
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        if inout is not None:
            model = inout.load_ply(model_path, vertex_scale=vertex_scale)
            pts = model["pts"].astype(np.float64)
        else:
            pts = _load_ply_xyz_binary_little_endian(model_path).astype(np.float64) * float(vertex_scale)
        fps_dict[str(obj_id)] = {}
        get_fps = _get_fps_and_center_fallback
        fps_dict[str(obj_id)]["fps4_and_center"] = get_fps(pts, num_fps=4, init_center=True)
        fps_dict[str(obj_id)]["fps8_and_center"] = get_fps(pts, num_fps=8, init_center=True)
        fps_dict[str(obj_id)]["fps12_and_center"] = get_fps(pts, num_fps=12, init_center=True)
        fps_dict[str(obj_id)]["fps16_and_center"] = get_fps(pts, num_fps=16, init_center=True)
        fps_dict[str(obj_id)]["fps20_and_center"] = get_fps(pts, num_fps=20, init_center=True)
        fps_dict[str(obj_id)]["fps32_and_center"] = get_fps(pts, num_fps=32, init_center=True)
        fps_dict[str(obj_id)]["fps64_and_center"] = get_fps(pts, num_fps=64, init_center=True)
        fps_dict[str(obj_id)]["fps128_and_center"] = get_fps(pts, num_fps=128, init_center=True)
        fps_dict[str(obj_id)]["fps256_and_center"] = get_fps(pts, num_fps=256, init_center=True)

    save_path = osp.join(model_dir, "fps_points.pkl")
    if mmcv is not None:
        mmcv.dump(fps_dict, save_path)
    else:
        with open(save_path, "wb") as f:
            pickle.dump(fps_dict, f, protocol=4)
    print(f"saved to {save_path}")


if __name__ == "__main__":
    main()
