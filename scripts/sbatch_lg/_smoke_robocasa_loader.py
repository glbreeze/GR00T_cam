"""Smoke test: build the RoboCasa baseline dataset exactly as gr00t_finetune.py
does and pull a few samples (incl. video) to prove the video_path fix works.
Run on a GPU compute node (importing gr00t touches CUDA at import time)."""

import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config

DATASET = "/scratch/lg154/cache/lerobot/robocasa24/all24_human_camaware_gr00t"

cfg = load_data_config("robocasa")
# baseline: no camera params, deterministic resize (matches train_robocasa_baseline)
cfg.use_camera_params = False
cfg.disable_geometric_augs = True

ds = LeRobotSingleDataset(
    dataset_path=DATASET,
    modality_configs=cfg.modality_config(),
    transforms=cfg.transform(),
    embodiment_tag=EmbodimentTag("new_embodiment"),
    video_backend="decord",
)

print(f"[ok] dataset built: {len(ds)} steps")

# Sample a few indices, including the first (chunk 0 / ep 0) that crashed before.
idxs = [0, len(ds) // 2, len(ds) - 1]
for i in idxs:
    s = ds[i]
    keys = sorted(s.keys())
    print(f"\n[sample {i}] keys={keys}")
    for k in keys:
        v = s[k]
        if isinstance(v, np.ndarray):
            print(f"    {k:40s} {str(v.shape):20s} {v.dtype}")
        else:
            print(f"    {k:40s} {type(v).__name__}")

print("\n[PASS] loaded {} samples incl. video, no video_path_pattern error".format(len(idxs)))
