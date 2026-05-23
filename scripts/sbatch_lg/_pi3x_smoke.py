"""End-to-end smoke test for the pi3x distillation wiring.

Goal: construct the CamVLA model with --use-pi3x-distill, pull one batch from
CameraAwareLeRobotDataset with pi3x_root set, run one forward+backward, print
the loss + aux loss. No optimizer step, no checkpoint save.
"""

import os
import sys

import torch

from gr00t.data.dataset_camvla import CameraAwareLeRobotDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator

DATASET_PATH = "/scratch/lg154/cache/lerobot/glbreeze/libero_cam_v2_gr00t"
PI3X_ROOT = "/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2"
DEVICE = "cuda"


def main() -> int:
    print("=== building data config ===")
    dc = load_data_config("libero")
    dc.use_camera_params = True
    dc.disable_geometric_augs = True
    modality_configs = dc.modality_config()
    transforms = dc.transform()

    print("=== building dataset ===")
    ds = CameraAwareLeRobotDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend="decord",
        pi3x_root=PI3X_ROOT,
    )
    print(f"  len(ds) = {len(ds)}")
    print(f"  has_camera_params = {ds.has_camera_params}")
    print(f"  has_pi3x_targets  = {ds.has_pi3x_targets}")
    print(f"  available_pi3x_subdirs = {ds._available_pi3x_subdirs}")

    print("=== pulling one sample ===")
    sample = ds[0]
    print(f"  sample keys: {sorted(sample.keys())}")
    for k in sample:
        if k.startswith("pi3x.") or k.startswith("camera."):
            v = sample[k]
            shp = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            print(f"  {k}: {shp}")

    # Pick which geometry modules to exercise via $PI3X_SMOKE_CONFIG.
    # Used to bisect which module is responsible when the loss NaNs.
    cfg = os.environ.get("PI3X_SMOKE_CONFIG", "full")
    cfg_flags = {
        "ray":       dict(use_ray_embed=True),
        "simple":    dict(cross_view_type="simple"),
        "standard":  dict(cross_view_type="standard", cross_view_aa_order="fg"),
        "prope":     dict(cross_view_type="standard", pose_enc_type="prope",
                          cross_view_aa_order="fg", cross_view_prope_layer_idx=(0,)),
        "pi3x":      dict(use_pi3x_distill=True),
        "full":      dict(use_ray_embed=True,
                          cross_view_type="standard", pose_enc_type="prope",
                          cross_view_aa_order="fg", cross_view_prope_layer_idx=(0,),
                          use_pi3x_distill=True),
    }[cfg]
    print(f"=== building model (config={cfg!r}, flags={cfg_flags}) ===")
    model = GR00T_N1_5.from_pretrained(
        "nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=False,
        use_camvla_model=True,
        pi3x_loss_weight=1.0,
        pi3x_loss_type="pi3x_local_pointmap",
        **cfg_flags,
    )
    # FlashAttention requires bf16/fp16, and the production training script
    # uses bf16 mixed precision via TrainingArguments. Match that here.
    model = model.to(device=DEVICE, dtype=torch.bfloat16)
    model.train()

    em = model.backbone.eagle_model
    has_ph = hasattr(em, "point_head")
    has_ray = hasattr(em, "ray_embed_module")
    has_cv = hasattr(em, "cross_view_fusion")
    print(f"  ray_embed_module present: {has_ray}")
    print(f"  cross_view_fusion present: {has_cv} ({type(getattr(em, 'cross_view_fusion', None)).__name__})")
    print(f"  point_head present: {has_ph}")
    if has_ph:
        n_params = sum(p.numel() for p in em.point_head.parameters())
        print(f"  point_head total params: {n_params:,}")

    print("=== collating batch (size 2) ===")
    collator = DefaultDataCollator()
    batch = collator([ds[0], ds[1]])
    for k, v in batch.items():
        if k.startswith("pi3x.") or k.startswith("camera."):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # ---- Probe PointHead numerics before forward (when enabled) ----
    ph = getattr(model.backbone.eagle_model, "point_head", None)
    if ph is not None:
        print("=== point_head param stats (post-from_pretrained, post-bf16-cast) ===")
        for name, p in ph.named_parameters():
            v = p.detach().float()
            print(
                f"  {name}: shape={tuple(p.shape)} "
                f"min={v.min().item():.3e} max={v.max().item():.3e} "
                f"abs_mean={v.abs().mean().item():.3e}"
            )

    print("=== forward + backward ===")
    out = model(batch)
    print(f"  output keys: {sorted(out.keys())}")
    loss = out["loss"]
    aux = out.get("aux_loss_pi3x")
    print(f"  loss: {loss.item():.6f}  requires_grad={loss.requires_grad}  grad_fn={loss.grad_fn}")
    if aux is None:
        print("  aux_loss_pi3x: MISSING (distill not wired)")
        return 1
    print(f"  aux_loss_pi3x: {float(aux):.6f}  requires_grad={aux.requires_grad}  grad_fn={aux.grad_fn}")

    if not torch.isfinite(loss):
        print("  ERROR: loss is not finite")
        return 1

    # Also probe whether the action_head projector receives a non-zero grad —
    # serves as a sanity check that backward() runs end-to-end and isn't
    # silently no-op'ing.
    ah_param = next(
        (p for n, p in model.action_head.named_parameters() if "projector" in n and p.requires_grad),
        None,
    )
    loss.backward()
    print("  backward OK")

    # Bucket gradient stats by geometry submodule so we can see which previously-
    # dead modules now receive a signal.
    buckets = {
        "ray_embed_module": [],
        "cross_view_fusion": [],
        "point_head": [],
        "action_head": [],
        "other_backbone": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        s = p.grad.float().abs().sum().item()
        if "ray_embed_module" in name:
            buckets["ray_embed_module"].append((name, s))
        elif "cross_view_fusion" in name:
            buckets["cross_view_fusion"].append((name, s))
        elif "point_head" in name:
            buckets["point_head"].append((name, s))
        elif name.startswith("action_head"):
            buckets["action_head"].append((name, s))
        else:
            buckets["other_backbone"].append((name, s))

    print("  gradient flow by submodule (count_nonzero / total, max |grad|_sum):")
    for bname, items in buckets.items():
        if not items:
            print(f"    {bname}: (no trainable params)")
            continue
        nz = [(n, s) for n, s in items if s > 0]
        max_s = max((s for _, s in items), default=0.0)
        print(f"    {bname}: {len(nz)}/{len(items)} nonzero, max={max_s:.4e}")
        # Show one example nonzero entry for each bucket
        if nz:
            n, s = nz[0]
            print(f"      e.g. {n}: |grad|_sum={s:.4e}")

    # PointHead gradient inspection. After step 0, only `linear_out` should
    # see a non-zero gradient (the trunk's input flows through ``linear_out``,
    # whose weight is zero, so the trunk's downstream grad is zero on step 0;
    # ``linear_out`` itself gets a non-zero grad from `trunk_out^T @ d_loss`).
    ph = model.backbone.eagle_model.point_head
    print("  point_head grad summary:")
    nonzero_names = []
    none_names = []
    for name, p in ph.named_parameters():
        if p.grad is None:
            none_names.append(name)
            continue
        g_abs_sum = p.grad.float().abs().sum().item()
        if g_abs_sum > 0:
            nonzero_names.append((name, g_abs_sum))
    for name, s in nonzero_names[:8]:
        print(f"    nonzero  {name}: |grad|_sum={s:.4e}")
    print(f"    {len(nonzero_names)} params with nonzero grad")
    print(f"    {len(none_names)} params with grad=None (first 3): {none_names[:3]}")

    print("=== SMOKE TEST OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
