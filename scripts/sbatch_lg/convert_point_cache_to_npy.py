#!/usr/bin/env python
"""Re-pack a compressed per-episode point-target cache into uncompressed per-array
``.npy`` files so the dataloader can mmap a single frame instead of decompressing
the whole episode on every sample.

Source layout (openpi-flavor, compressed ``np.savez_compressed``):
    {src}/{cam}/episode_{NNNNNN}.npz   with keys {xy, log_z, conf}, each (T, H, W, *)

Destination layout (uncompressed, mmap-friendly):
    {dst}/{cam}/episode_{NNNNNN}/xy.npy
                                /log_z.npy
                                /conf.npy

dtype is preserved (the caches are float16). The destination is numerically
identical to the source — only the on-disk encoding changes. ``CameraAwareLeRobotDataset``
auto-detects the .npy layout (see ``_load_cam_frame``) and falls back to .npz.

Idempotent: an episode whose 3 .npy files already exist (and are non-empty) is
skipped unless --overwrite is passed.

Example:
    python convert_point_cache_to_npy.py \
        --src /scratch/yz11445/.cache/openpi/gt_point_targets_grid224_camfix/robocasa24_all24_human_camaware \
        --dst /scratch/lg154/cache/point_targets_npy/robocasa24_all24_human_camaware \
        --subdirs base wrist --workers 32
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ARRAY_KEYS = ("xy", "log_z", "conf")


def _convert_episode(npz_path: pathlib.Path, out_dir: pathlib.Path, overwrite: bool) -> tuple[str, str]:
    """Convert one episode .npz -> a dir of .npy. Returns (status, name)."""
    name = npz_path.stem  # episode_000000
    target = out_dir / name
    done = all((target / f"{k}.npy").exists() and (target / f"{k}.npy").stat().st_size > 0 for k in ARRAY_KEYS)
    if done and not overwrite:
        return ("skip", name)
    target.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path) as f:
        for k in ARRAY_KEYS:
            arr = np.asarray(f[k])  # preserve dtype (float16)
            tmp = target / f"{k}.npy.tmp"
            # Pass a file handle so np.save does NOT re-append ".npy" to the name.
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            tmp.replace(target / f"{k}.npy")  # atomic: never leave a half-written .npy
    return ("ok", name)


def convert_cam(src_cam: pathlib.Path, dst_cam: pathlib.Path, workers: int, overwrite: bool) -> None:
    episodes = sorted(src_cam.glob("episode_*.npz"))
    if not episodes:
        print(f"[WARN] no episode_*.npz under {src_cam}", file=sys.stderr)
        return
    dst_cam.mkdir(parents=True, exist_ok=True)
    print(f"[{src_cam.name}] {len(episodes)} episodes -> {dst_cam}", flush=True)
    n_ok = n_skip = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_convert_episode, ep, dst_cam, overwrite) for ep in episodes]
        for i, fut in enumerate(as_completed(futs), 1):
            status, _ = fut.result()
            n_ok += status == "ok"
            n_skip += status == "skip"
            if i % 100 == 0 or i == len(episodes):
                print(f"  [{src_cam.name}] {i}/{len(episodes)} (ok={n_ok} skip={n_skip})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=pathlib.Path, help="source cache root (contains cam subdirs)")
    ap.add_argument("--dst", required=True, type=pathlib.Path, help="destination root for .npy layout")
    ap.add_argument("--subdirs", nargs="+", required=True, help="cam subdirs to convert, e.g. base wrist")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true", help="re-convert episodes even if .npy already exist")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"src not found: {args.src}")
    for sub in args.subdirs:
        src_cam = args.src / sub
        if not src_cam.is_dir():
            sys.exit(f"missing cam subdir: {src_cam}")
        convert_cam(src_cam, args.dst / sub, args.workers, args.overwrite)
    print(f"DONE. Re-pointed cache root: {args.dst}", flush=True)


if __name__ == "__main__":
    main()
