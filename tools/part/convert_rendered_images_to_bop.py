import argparse
import json
import math
import os
import os.path as osp
import random
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PoseDelta:
    kind: str
    axis: str
    value: float


def parse_pose_delta_from_stem(stem: str) -> Optional[PoseDelta]:
    m = re.search(r"(平移|旋转)\s*([xyzXYZ])\s*[\(（]\s*([+-]?\d+(?:\.\d+)?)\s*[\)）]", stem)
    if not m:
        return None
    kind_raw, axis_raw, value_raw = m.group(1), m.group(2), m.group(3)
    kind = "trans" if kind_raw == "平移" else "rot"
    axis = axis_raw.lower()
    value = float(value_raw)
    return PoseDelta(kind=kind, axis=axis, value=value)


def rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def delta_to_pose(
    delta: PoseDelta,
    t0_mm: np.ndarray,
    apply_rotation_left: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    t = t0_mm.astype(np.float32).copy()
    R0 = np.eye(3, dtype=np.float32)
    if delta.kind == "trans":
        idx = {"x": 0, "y": 1, "z": 2}[delta.axis]
        t[idx] += float(delta.value)
        return R0, t
    rad = math.radians(delta.value)
    if delta.axis == "x":
        Rd = rot_x(rad)
    elif delta.axis == "y":
        Rd = rot_y(rad)
    else:
        Rd = rot_z(rad)
    R = Rd @ R0 if apply_rotation_left else R0 @ Rd
    return R, t


def find_images(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                out.append(osp.join(dirpath, fn))
    out.sort()
    return out


def leaf_dir(path: str, root: str) -> str:
    rel = osp.relpath(osp.dirname(path), root)
    return rel.replace("\\", "/")


def load_gray_u8(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    return np.array(im, dtype=np.uint8)


def estimate_background_value(gray: np.ndarray) -> int:
    vals, counts = np.unique(gray, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def make_mask_from_gray(gray: np.ndarray, bg_value: int, thr: int) -> np.ndarray:
    diff = np.abs(gray.astype(np.int16) - int(bg_value))
    mask = (diff > int(thr)).astype(np.uint8) * 255
    return mask


def bbox_xywh_from_mask(mask_u8: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]


def split_indices(n: int, test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_test = int(round(n * float(test_ratio)))
    test = sorted(idxs[:n_test])
    train = sorted(idxs[n_test:])
    return train, test


def load_ply_xyz_binary_little_endian(path: str) -> np.ndarray:
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
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    return xyz


def render_depth_from_vertices(
    verts_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = verts_cam[:, 0]
    y = verts_cam[:, 1]
    z = verts_cam[:, 2]
    valid = z > 1e-6
    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = np.rint(fx * (x / z) + cx).astype(np.int32)
    v = np.rint(fy * (y / z) + cy).astype(np.int32)
    inside = (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    u = u[inside]
    v = v[inside]
    z = z[inside].astype(np.float32)

    depth_flat = np.full((int(height) * int(width),), np.inf, dtype=np.float32)
    lin = v.astype(np.int64) * int(width) + u.astype(np.int64)
    np.minimum.at(depth_flat, lin, z)
    depth = depth_flat.reshape(int(height), int(width))
    depth[~np.isfinite(depth)] = 0.0
    return depth


def depth_to_xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    invK = np.linalg.inv(K.astype(np.float64))
    ys, xs = np.where(depth > 0)
    xyz = np.zeros((h, w, 3), dtype=np.float32)
    if len(xs) == 0:
        return xyz
    pix = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones_like(xs, dtype=np.float64)], axis=0)  # 3xN
    rays = (invK @ pix).T.astype(np.float32)  # Nx3
    z = depth[ys, xs].astype(np.float32)
    xyz_vals = rays * z[:, None]
    xyz[ys, xs, :] = xyz_vals
    return xyz


def dump_pickle(path: str, data: Dict) -> None:
    ensure_dir(osp.dirname(path))
    import pickle

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def dump_json(path: str, data: Dict) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_png_u8(path: str, arr: np.ndarray) -> None:
    ensure_dir(osp.dirname(path))
    Image.fromarray(arr.astype(np.uint8)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, type=str)
    ap.add_argument("--out-root", default=osp.join("datasets", "BOP_DATASETS", "part"), type=str)
    ap.add_argument("--split", default="train", choices=["train", "test"], type=str)
    ap.add_argument("--scene-mode", default="by_folder", choices=["by_folder", "single"], type=str)
    ap.add_argument("--max-images", default=-1, type=int)
    ap.add_argument("--test-ratio", default=0.1, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--k", default="636.82,0,635.21,0,637.21,362.37,0,0,1", type=str)
    ap.add_argument("--t0-mm", default="0,0,160", type=str)
    ap.add_argument("--mask-thr", default=8, type=int)
    ap.add_argument("--apply-rotation-left", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--gen-xyz", action="store_true")
    ap.add_argument("--model-ply", default="", type=str)
    ap.add_argument("--model-vertex-scale", default=1.0, type=float)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    k_vals = [float(x) for x in args.k.split(",")]
    if len(k_vals) != 9:
        raise ValueError("--k must contain 9 comma-separated numbers")
    K = np.array(k_vals, dtype=np.float32).reshape(3, 3)

    t0_vals = [float(x) for x in args.t0_mm.split(",")]
    if len(t0_vals) != 3:
        raise ValueError("--t0-mm must contain 3 comma-separated numbers")
    t0 = np.array(t0_vals, dtype=np.float32)

    model_xyz = None
    if args.gen_xyz:
        model_ply = args.model_ply or osp.join(args.out_root, "models", "obj_000001.ply")
        if not osp.exists(model_ply):
            raise RuntimeError(f"model ply not found: {model_ply}")
        model_xyz = load_ply_xyz_binary_little_endian(model_ply) * float(args.model_vertex_scale)

    imgs = find_images(args.src_root)
    if len(imgs) == 0:
        raise RuntimeError("no .png found")
    if int(args.max_images) > 0:
        imgs = imgs[: int(args.max_images)]

    if args.split == "train":
        train_idx, test_idx = split_indices(len(imgs), args.test_ratio, args.seed)
        sel = train_idx
    else:
        train_idx, test_idx = split_indices(len(imgs), args.test_ratio, args.seed)
        sel = test_idx

    if args.scene_mode == "single":
        scene_keys = {"000000": [imgs[i] for i in sel]}
    else:
        groups: Dict[str, List[str]] = {}
        for i in sel:
            p = imgs[i]
            key = leaf_dir(p, args.src_root)
            groups.setdefault(key, []).append(p)
        sorted_keys = sorted(groups.keys())
        scene_keys = {f"{si:06d}": groups[k] for si, k in enumerate(sorted_keys)}

    out_split_root = osp.join(args.out_root, args.split)
    ensure_dir(out_split_root)

    all_scene_im_lines = []

    for scene_id, scene_imgs in scene_keys.items():
        scene_root = osp.join(out_split_root, scene_id)
        rgb_dir = osp.join(scene_root, "rgb")
        mask_dir = osp.join(scene_root, "mask")
        mask_visib_dir = osp.join(scene_root, "mask_visib")
        ensure_dir(rgb_dir)
        ensure_dir(mask_dir)
        ensure_dir(mask_visib_dir)

        scene_gt: Dict[str, List[Dict]] = {}
        scene_cam: Dict[str, Dict] = {}
        scene_gt_info: Dict[str, List[Dict]] = {}
        xyz_scene_root = osp.join(out_split_root, "xyz_crop", scene_id)

        for im_id, src_path in enumerate(scene_imgs):
            stem = osp.splitext(osp.basename(src_path))[0]
            delta = parse_pose_delta_from_stem(stem)
            if delta is None:
                raise RuntimeError(f"cannot parse pose delta from filename: {src_path}")
            R, t_mm = delta_to_pose(delta, t0_mm=t0, apply_rotation_left=bool(args.apply_rotation_left))

            im_key = str(im_id)
            dst_rgb = osp.join(rgb_dir, f"{im_id:06d}.png")
            dst_mask = osp.join(mask_dir, f"{im_id:06d}_000000.png")
            dst_mask_vis = osp.join(mask_visib_dir, f"{im_id:06d}_000000.png")

            gray = load_gray_u8(src_path)
            bg = estimate_background_value(gray)
            mask = make_mask_from_gray(gray, bg_value=bg, thr=int(args.mask_thr))
            bbox_xywh = bbox_xywh_from_mask(mask)
            if bbox_xywh is None:
                bbox_xywh = [0, 0, int(gray.shape[1]), int(gray.shape[0])]

            px_vis = int((mask > 0).sum())
            info = {
                "bbox_obj": bbox_xywh,
                "bbox_visib": bbox_xywh,
                "px_count_all": px_vis,
                "px_count_valid": px_vis,
                "px_count_visib": px_vis,
                "visib_fract": 1.0,
            }

            ann = {
                "obj_id": 1,
                "cam_R_m2c": R.reshape(-1).tolist(),
                "cam_t_m2c": t_mm.reshape(-1).tolist(),
            }

            cam = {
                "cam_K": K.reshape(-1).tolist(),
                "depth_scale": 1000.0,
            }

            scene_gt[im_key] = [ann]
            scene_cam[im_key] = cam
            scene_gt_info[im_key] = [info]

            if args.dry_run:
                continue
            shutil.copy2(src_path, dst_rgb)
            save_png_u8(dst_mask, mask)
            save_png_u8(dst_mask_vis, mask)
            if args.gen_xyz:
                x1, y1, bw, bh = bbox_xywh
                x2 = x1 + bw - 1
                y2 = y1 + bh - 1
                h, w = gray.shape[:2]
                x1 = max(min(int(x1), w - 1), 0)
                y1 = max(min(int(y1), h - 1), 0)
                x2 = max(min(int(x2), w - 1), 0)
                y2 = max(min(int(y2), h - 1), 0)

                Rm2c = R.astype(np.float32)
                tm2c_m = (t_mm.astype(np.float32) / 1000.0).reshape(1, 3)
                verts_cam = (model_xyz @ Rm2c.T) + tm2c_m
                depth = render_depth_from_vertices(verts_cam, K=K, width=w, height=h)
                xyz = depth_to_xyz(depth, K=K)
                xyz_crop = xyz[y1 : y2 + 1, x1 : x2 + 1, :].astype(np.float16)
                xyz_path = osp.join(xyz_scene_root, f"{im_id:06d}_000000-xyz.pkl")
                dump_pickle(xyz_path, {"xyz_crop": xyz_crop, "xyxy": [x1, y1, x2, y2]})

            all_scene_im_lines.append(f"{scene_id}/{im_id:06d}")

        dump_json(osp.join(scene_root, "scene_gt.json"), scene_gt)
        dump_json(osp.join(scene_root, "scene_camera.json"), scene_cam)
        dump_json(osp.join(scene_root, "scene_gt_info.json"), scene_gt_info)

    image_sets_dir = osp.join(args.out_root, "image_sets")
    ensure_dir(image_sets_dir)
    idx_path = osp.join(image_sets_dir, f"{args.split}.txt")
    if not args.dry_run:
        with open(idx_path, "w", encoding="utf-8") as f:
            for line in all_scene_im_lines:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
