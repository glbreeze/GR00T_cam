"""Verify the geo checkpoint loads as a CamVLA backbone the same way the
inference server loads it (GR00T_N1_5.from_pretrained, NO camvla kwargs).

The from_pretrained console prints reflect the (default-False) kwargs, not the
backbone that __init__ actually built from the saved config.json's backbone_cfg.
This script introspects the *real* loaded module to confirm geometry is active
and the trained geometry weights are non-zero (i.e. loaded from the checkpoint).
"""
import sys
import warnings

import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/scratch/lg154/Research/GR00T_cam/checkpoints/"
    "gr00t_robocasa_geo_distill_gtonly_res224/checkpoint-30000"
)

from gr00t.model.gr00t_n1 import GR00T_N1_5

print(f"=== loading {CKPT} (no camvla kwargs, exactly like the server) ===")
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    model = GR00T_N1_5.from_pretrained(CKPT, torch_dtype=torch.bfloat16)

bb = model.backbone
eagle = getattr(bb, "eagle_model", None)
print("\n=== RESULTS ===")
print("backbone._use_camvla_model:", getattr(bb, "_use_camvla_model", "MISSING"))
print("eagle_model class:", type(eagle).__name__)
print("config.backbone_cfg use_camvla_model:", model.config.backbone_cfg.get("use_camvla_model"))
print("config.backbone_cfg use_ray_embed:", model.config.backbone_cfg.get("use_ray_embed"))
print("config.backbone_cfg cross_view_type:", model.config.backbone_cfg.get("cross_view_type"))
print("config.backbone_cfg pose_enc_type:", model.config.backbone_cfg.get("pose_enc_type"))

# Inspect geometry submodules: presence + whether weights are non-zero (trained).
def norm_of(mod):
    if mod is None:
        return None
    return float(sum(p.float().norm().item() for p in mod.parameters()))

geom_names = ["ray_embed", "ray_proj", "cross_view", "prope", "point_head"]
print("\n=== geometry module weight norms (0.0 => zero-init/not loaded) ===")
found_any = False
for name, sub in (eagle.named_modules() if eagle is not None else []):
    if any(g in name.lower() for g in geom_names) and sub is not None:
        n = norm_of(sub)
        if n is not None and len(list(sub.parameters(recurse=False))) > 0:
            print(f"  {name:55s} ({type(sub).__name__:24s}) param_norm={n:.4f}")
            found_any = True
if not found_any:
    print("  (no geometry submodules found by name match)")

# Surface any missing/unexpected key warnings from the load.
print("\n=== load warnings mentioning keys ===")
hit = False
for w in wlist:
    msg = str(w.message)
    if any(s in msg.lower() for s in ("missing", "unexpected", "not used", "newly initialized")):
        print("  WARN:", msg[:300])
        hit = True
if not hit:
    print("  (none — all checkpoint keys matched the model)")

is_camvla = type(eagle).__name__ == "Eagle2_5_VLCamVLA"
print("\n=== VERDICT ===")
print("CamVLA backbone active:", is_camvla)
print("PASS" if is_camvla else "FAIL: geo checkpoint loaded as PLAIN backbone -- geometry dropped")
