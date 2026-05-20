
 deactivate                                                          # drop openpi venv
  conda activate gr00t
  export WANDB_API_KEY="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"
  python scripts/gr00t_finetune.py \
    --dataset-path /home/asus/Research/datasets/libero_cam_v2_gr00t \
    --data-config libero \
    --video-backend decord \
    --num-gpus 1 \
    --no-tune_diffusion_model \
    --use-camvla-model

Four supported invocations

# 1) Baseline (already validated earlier — byte-for-byte unchanged)
python scripts/gr00t_finetune.py --dataset-path ... --data-config libero --video-backend decord

# 2) Baseline + CamVLA model subclass (model hooks ready, no data changes yet)
python scripts/gr00t_finetune.py ... --use-camvla-model

# 3) Full camera-aware (random crop active, K stays in sync with cropped pixels)
python scripts/gr00t_finetune.py ... --use-camera-params --use-camvla-model

# 4) Pi3x distillation mode (deterministic resize, color jitter only — cache-aligned)
python scripts/gr00t_finetune.py ... --use-camera-params --use-camvla-model --disable-geometric-augs

Each sample now carries:
- camera.agent_intrinsic / camera.wrist_intrinsic: [T=1, 3, 3], updated through the transform chain
- camera.agent_extrinsic / camera.wrist_extrinsic: [T=1, 4, 4], pass-through (no 2D-aug update)


  if use_ray_embed:        vit += RayEmbed(K)
  if cross_view_type == "simple":     vit = SimpleCrossView(vit)
  elif cross_view_type == "standard": vit = StandardCrossView(vit, E if prope else None, K if prope else None)

    Validation rules at config time:
  - pose_enc_type=="prope" requires cross_view_type=="standard" (raise otherwise — matches openpi's silent-drop but louder)
  - cross_view_prope_layer_idx requires pose_enc_type=="prope"
  - All entries of cross_view_prope_layer_idx must be valid global-block indices given cross_view_aa_order



  # Baseline-equivalent runs
  --use-camvla-model

  # Ray embedding only (openpi: ray_enc_type=true)
  --use-camvla-model --use-ray-embed

  # Simple cross-view (openpi: cross_view.type=simple)
  --use-camvla-model --cross-view-type simple

  # Standard cross-view, no pose (openpi: cross_view.type=standard, pose_enc_type=null)
  --use-camvla-model --cross-view-type standard --cross-view-aa-order fg

  # Standard cross-view + PRoPE after the only global block
  --use-camvla-model --cross-view-type standard --pose-enc-type prope \
      --cross-view-aa-order fg --cross-view-prope-layer-idx 0