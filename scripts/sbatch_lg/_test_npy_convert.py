"""Correctness check for the .npz->.npy re-cache + the loader's .npy path.

Converts a couple of episodes to a temp dir and verifies:
  1. every .npy frame is bit-identical to the source .npz frame, and
  2. the dataset's _load_cam_frame mmap path returns the same single frame.
Run on a compute node (decompresses ~2 episodes).
"""
import importlib.util
import pathlib
import tempfile

import numpy as np

SRC = pathlib.Path(
    "/scratch/yz11445/.cache/openpi/gt_point_targets_grid224_camfix/robocasa24_all24_human_camaware"
)
SUBDIRS = ("base", "wrist")
N_EP = 2

# import the conversion module in isolation (absolute path -> cwd-independent)
_REPO = pathlib.Path(__file__).resolve().parents[2]
conv = importlib.util.spec_from_file_location(
    "conv", _REPO / "scripts/sbatch_lg/convert_point_cache_to_npy.py")
convmod = importlib.util.module_from_spec(conv); conv.loader.exec_module(convmod)

# dataset_camvla imports gr00t.data.* — too heavy here; replicate _load_cam_frame
# by reading it back the same way the loader does (mmap one frame).


def load_cam_frame_npy(cam_dir, traj, idx):
    ep = f"episode_{int(traj):06d}"
    d = cam_dir / ep
    out = []
    for name in ("xy", "log_z", "conf"):
        arr = np.load(d / f"{name}.npy", mmap_mode="r")
        out.append(np.asarray(arr[idx], dtype=np.float32))
    return out


with tempfile.TemporaryDirectory(dir="/scratch/lg154") as tmp:
    dst = pathlib.Path(tmp) / "npy"
    for sub in SUBDIRS:
        for e in range(N_EP):
            npz = SRC / sub / f"episode_{e:06d}.npz"
            status, name = convmod._convert_episode(npz, dst / sub, overwrite=False)
            # full-array equality vs source
            with np.load(npz) as f:
                T = f["xy"].shape[0]
                for k in ("xy", "log_z", "conf"):
                    src = np.asarray(f[k])
                    got = np.load(dst / sub / name / f"{k}.npy")
                    assert src.shape == got.shape and src.dtype == got.dtype, (k, src.shape, got.shape, src.dtype, got.dtype)
                    assert np.array_equal(src, got), f"mismatch {sub}/{name}/{k}"
                # single-frame mmap read matches a few frames
                for idx in (0, T // 2, T - 1):
                    xy, lz, cf = load_cam_frame_npy(dst / sub, e, idx)
                    assert np.array_equal(xy, np.asarray(f["xy"][idx], dtype=np.float32))
                    assert np.array_equal(lz, np.asarray(f["log_z"][idx], dtype=np.float32))
                    assert np.array_equal(cf, np.asarray(f["conf"][idx], dtype=np.float32))
            print(f"  OK {sub}/{name}  status={status}  T={T}  shape={got.shape} dtype={got.dtype}")
print("PASS: .npy re-cache is bit-identical to source .npz; mmap single-frame read matches")
