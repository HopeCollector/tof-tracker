"""Deterministic inputs shared by correctness and timing experiments."""
from pathlib import Path
import hashlib
import math
import os

import numpy as np

from devtools.config import load_config
from devtools.scene import Pose, Scene

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / "configs/devtools.json")


def load_scene(count=500):
    folder = Path(os.environ.get("PYTORCH3D_EVAL_MAP_DIR", ROOT / "outputs/sensor_perf/maps"))
    path = folder / f"pillars-{count}.obj"
    if not path.is_file():
        raise FileNotFoundError(f"Existing experiment scene required: {path}")
    scene = Scene.from_obj(path, CONFIG, Pose((0, 0, 1.5)), (4, 0, 1.5), t_ref=12.5)
    return scene, {"map": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "pillars": count, "target_position_m": [4, 0, 1.5]}


def euler_quaternion(roll=0., pitch=0., yaw=0.):
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def make_poses(batch, mode="camera"):
    """The first B views are identical in every sweep; no repeated identical view."""
    result = []
    for i in range(batch):
        phase = (i % 1024) * (math.sqrt(5) - 1) * math.pi
        roll = math.radians(8) * math.sin(phase * 3)
        pitch = math.radians(6) * math.cos(phase * 2)
        yaw = math.radians(12) * math.sin(phase)
        if mode == "tof":
            yaw += (-math.pi/2, 0, math.pi/2, math.pi)[i % 4]
        result.append(Pose((0.15 * math.sin(phase), 0.10 * math.cos(phase),
                            1.5 + 0.08 * math.sin(phase * 2)), euler_quaternion(roll, pitch, yaw)))
    return result


def poses_tensor(poses, device="cuda"):
    import torch
    matrices = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    for i, pose in enumerate(poses):
        matrices[i, :3, :3] = pose.as_transform().rotation
        matrices[i, :3, 3] = pose.position_m
    return torch.as_tensor(matrices, dtype=torch.float32, device=device)


def depth_comparison(actual, expected, actual_id=None, expected_id=None):
    """Do not hide nonfinite or silhouette disagreements in a mean error."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    classify = lambda v: np.where(np.isfinite(v), 0, np.where(np.isposinf(v), 1, 2))
    a, e = classify(actual), classify(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    errors = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    out = {"pixels": int(actual.size), "category_order": ["finite", "+inf", "NaN"],
           "category_confusion_expected_rows": [[int(np.count_nonzero((e == i) & (a == j)))
                                                 for j in range(3)] for i in range(3)],
           "category_disagreements": int(np.count_nonzero(a != e)),
           "finite_common": int(errors.size),
           "finite_max_abs_m": float(errors.max()) if errors.size else None,
           "finite_p99_abs_m": float(np.quantile(errors, .99)) if errors.size else None,
           "finite_mean_abs_m": float(errors.mean()) if errors.size else None}
    if actual_id is not None:
        mismatch = np.asarray(actual_id) != np.asarray(expected_id)
        # Interior excludes any 8-neighbor identity boundary in either image.
        interior = np.ones(mismatch.shape, dtype=bool)
        interior[..., 0, :] = interior[..., -1, :] = False
        interior[..., :, 0] = interior[..., :, -1] = False
        for identities in (np.asarray(actual_id), np.asarray(expected_id)):
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                interior &= identities == np.roll(identities, (dy, dx), axis=(-2, -1))
        out.update(identity_disagreements=int(mismatch.sum()), interior_pixels=int(interior.sum()),
                   interior_identity_disagreements=int((mismatch & interior).sum()))
    return out
