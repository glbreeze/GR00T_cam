# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.dataset_camvla import CameraAwareLeRobotDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data configuration to use for training.
    Options:
    - Built-in configs: Use predefined config names like 'so100', 'fourier_gr1_arms_only', 'unitree_g1'.
    - External configs: Use 'module:ClassName' format to load custom configs from external files. e.g. 'my_dir.my_configs:RobotConfig'
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    use_camvla_model: bool = False
    """If True, use the CamVLA Eagle2.5 subclass that exposes hooks for geometry
    modules (ray embedding, cross-view attention, pi3x distillation). Identical
    to baseline until those hooks are populated."""

    use_ray_embed: bool = False
    """If True, add a per-patch ray embedding from K^-1 to vit_embeds.
    Pi3X-inspired, zero-initialised so disabling = no contribution.
    Requires --use-camvla-model and camera intrinsics in the batch."""

    cross_view_type: Literal["none", "simple", "standard"] = "none"
    """Cross-view fusion topology applied to vit_embeds. 'simple' is one
    bidirectional attention block across (V*N) tokens. 'standard' is a stack
    of frame/global blocks per --cross-view-aa-order, into which PRoPE
    pose-injection blocks can be interleaved. Requires --use-camvla-model."""

    pose_enc_type: Literal["null", "prope"] = "null"
    """How to inject camera pose. 'prope' interleaves PoseInjectBlocks inside
    the standard cross-view fusion stack. Requires --cross-view-type standard."""

    cross_view_aa_order: str = "fg"
    """Frame/global ordering for the standard cross-view fusion stack, e.g.
    'fg' = one frame block then one global block. Each char consumes one block."""

    cross_view_prope_layer_idx: tuple[int, ...] = ()
    """Indices (into the 'g' sub-sequence of cross_view_aa_order) after which
    a PoseInjectBlock is inserted. Only used when --pose-enc-type prope."""

    cross_view_rope_freq: float = 100.0
    """Frequency base for 2D RoPE in cross-view frame/global blocks and for
    X/Y RoPE inside PRoPE attention. Set <= 0 to disable RoPE in frame/global
    blocks (PoseInjectBlock still uses RoPE internally with the same base)."""

    use_camera_params: bool = False
    """If True, load camera intrinsics/extrinsics from the parquet (via
    CameraAwareLeRobotDataset) and route them through camera-aware video
    transforms that keep K consistent with the augmented pixel grid."""

    disable_geometric_augs: bool = False
    """If True, drop the random crop from the video pipeline. Use when training
    against a precomputed pi3x distillation cache (random crops would invalidate
    the cached targets). Color jitter and the deterministic resize-to-224 stay."""

    pi3x_root: str | None = None
    """Path to a pi3x target cache (one ``episode_NNNNNN.npz`` per episode per
    cam, e.g. ``.../pi3x_targets_224/libero_cam_v2``). Setting this enables the
    distillation aux loss; requires --use-camvla-model, --use-camera-params, and
    --disable-geometric-augs (the cache is tied to a deterministic 224x224 resize)."""

    gt_point_root: str | None = None
    """Path to a ground-truth pointmap cache (same on-disk layout as --pi3x-root:
    ``episode_NNNNNN.npz`` per episode per cam under ``{root}/{agent,wrist}/``,
    each with ``xy`` / ``log_z`` / ``conf`` arrays). Mirrors openpi's
    ``gt_point_targets_root`` and the ``*_gtonly`` configs: supervises the
    PointHead with ground-truth pointmaps derived from the simulator. Set this
    alone for GT-only supervision, or alongside --pi3x-root for dual-loss
    training (the two are mixed by --point-target-gt-weight). Shares the same
    PointHead loss path and all --pi3x-loss-* knobs. Requires --use-camvla-model,
    --use-camera-params, and --disable-geometric-augs."""

    pi3x_cam_subdirs: tuple[str, ...] | None = None
    """Per-camera subdir names under --pi3x-root / --gt-point-root, in video order
    (the first must align with video.image, the second with video.wrist_image).
    Defaults to CameraAwareLeRobotDataset.PI3X_CAM_SUBDIRS = ("agent", "wrist"),
    which matches the Libero caches. RoboCasa's GT cache names the front view
    "base", so pass ``--pi3x-cam-subdirs base wrist`` there — otherwise only the
    matching subdir is detected and the front-camera targets are silently dropped
    (a non-matching name is skipped without error). No effect unless --pi3x-root
    or --gt-point-root is set."""

    point_target_gt_weight: float = 0.5
    """Dual-loss mix weight when BOTH --pi3x-root and --gt-point-root are set:
    aux_loss = w * L(pred, gt) + (1 - w) * L(pred, pi3x), with w this value in
    [0, 1]. Mirrors openpi's ``point_target_gt_ratio`` in ``dual_loss`` mode.
    Ignored when only one point-target source is supplied."""

    pi3x_loss_weight: float = 1.0
    """Outer coefficient applied to the pi3x aux loss before adding it to the
    action loss. Openpi uses 1.0 in stage 1, 0.05 in stage 2."""

    pi3x_loss_type: Literal["pi3x_local_pointmap", "legacy_conf_mse"] = "pi3x_local_pointmap"
    """Which pi3x loss to apply. 'pi3x_local_pointmap' (default) matches openpi's
    stage-1/2 training. 'legacy_conf_mse' is a simpler hard-confidence-gated L2
    that skips the per-sample scale alignment."""

    point_head_output_resolution: Literal[16, 224] = 16
    """PointHead output resolution. 16 (default): per-patch (16x16) prediction via
    a Linear head, paired with avg-pooled 16x16 targets. 224: full-resolution
    prediction via a Pi3X-style ConvHead upsampler, paired with un-pooled 224x224
    targets — matches openpi's ``aux_point_head.output_resolution=224``. Must agree
    with the on-disk target resolution; the dataset pools (16) or keeps full-res
    (224) to match. No effect unless --pi3x-root or --gt-point-root is set."""

    pi3x_ray_loss_weight: float = 1.0
    """Ray-direction (xy) loss weight inside pi3x_local_pointmap."""

    pi3x_depth_loss_weight: float = 1.0
    """Depth (z) loss weight inside pi3x_local_pointmap."""

    pi3x_depth_weighting: Literal["pi3x_inverse", "uniform"] = "pi3x_inverse"
    """Depth-weighting scheme inside pi3x_local_pointmap. 'pi3x_inverse' down-
    weights far points (matches openpi); 'uniform' treats all valid points equally."""

    action_loss_weight: float = 1.0
    """Multiplier on the action (flow-matching) loss before adding the pi3x aux
    loss. Two-stage training mirrors openpi: 0.1 in stage 1 (geometry warmup -
    aux dominates and still trickles gradient back through the frozen action
    path into cross_view_fusion), 1.0 in stage 2 (action-focused fine-tune)."""

    trainable_prefixes: tuple[str, ...] = ()
    """If non-empty, freeze every parameter whose name does NOT start with one
    of these prefixes (matched against ``model.named_parameters()``). Applied
    AFTER tune_visual/tune_llm/tune_projector/tune_diffusion_model, overriding
    them. Used for stage-1 geometry warmup, e.g.
    ``--trainable-prefixes backbone.eagle_model.ray_embed_module
    backbone.eagle_model.cross_view_fusion backbone.eagle_model.point_head``.
    Mirrors openpi's ``trainable_prefixes``."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    llm_learning_rate: float | None = None
    """Optional separate LR for the LLM backbone (``backbone.eagle_model.language_model.*``).
    ``None`` (default) = single LR for all params (baseline-identical). Set this
    when unfreezing the LLM (``--tune-llm``) to keep the proven action-head/geometry
    LR on ``--learning-rate`` while giving the pretrained LLM a gentler LR (e.g.
    ``--learning-rate 1e-4 --llm-learning-rate 2e-5``). Both groups share the same
    scheduler/warmup; only the peak (base) LR differs per group."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lr_scheduler_type: str = "cosine"
    """HuggingFace LR-scheduler family. Default ``cosine`` decays peak_lr -> 0
    at max_steps. Use ``cosine_with_min_lr`` to floor the schedule at a fraction
    of the peak LR (set ``--min-lr-rate`` accordingly). openpi's
    CosineDecaySchedule corresponds to ``cosine_with_min_lr`` + ``min_lr_rate=0.1``."""

    min_lr_rate: float = 0.0
    """Floor of the cosine schedule expressed as a fraction of the peak LR.
    Only applied when ``--lr-scheduler-type cosine_with_min_lr`` is set; ignored
    by every other scheduler (which is HF's behavior). Defaults to 0.0 so the
    plain ``cosine`` path is byte-for-byte unchanged."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    gradient_checkpointing: bool = False
    """Trade compute for memory by recomputing activations in the backward pass.
    Off by default (baseline-identical). Enable when unfreezing the LLM/vision
    (``--tune-llm`` / ``--tune-visual``) to fit larger batches on one GPU."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""


#####################################################################################
# Helper functions
#####################################################################################


def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    """
    Copy weights with partial dimension matching for action_dim changes.
    NOTE(Youliang): this is a very experimental implementation to handle action_dim changes. TODO: improve this.
    """
    total_params = copied_params = random_params = 0

    for key, old_tensor in old_dict.items():
        if key not in new_dict:
            continue

        new_tensor = new_dict[key]
        total_params += new_tensor.numel()

        if old_tensor.shape == new_tensor.shape:
            # Same shape: direct copy
            new_tensor.copy_(old_tensor)
            copied_params += new_tensor.numel()
        elif "action_encoder" in key and "W1.weight" in key:
            # Input dimension change: copy [:, :old_dim]
            new_tensor[:, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        elif "action_decoder" in key and ("weight" in key or "bias" in key):
            # Output dimension change: copy first old_dim elements of last dimension
            if old_tensor.dim() == 1:
                new_tensor[:old_dim] = old_tensor
            elif old_tensor.dim() == 2:
                new_tensor[:, :old_dim] = old_tensor
            elif old_tensor.dim() == 3:
                new_tensor[:, :, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        else:
            # Incompatible shape: keep random initialization
            random_params += new_tensor.numel()

    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    print(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random ({random_percentage:.1f}% randomly initialized)"
    )
    print(f"Action dimensions {old_dim+1}-{new_dim} will be learned from scratch")
    return new_dict


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # Point-head supervision can draw from two caches: pi3x teacher predictions
    # (--pi3x-root, distillation) and/or ground-truth pointmaps from the
    # simulator (--gt-point-root). Setting both enables dual-loss training
    # (mixed by --point-target-gt-weight). Either source requires the rest of
    # the camera-aware stack: CamVLA backbone (to host the PointHead), parquet
    # camera params (so K is known to the model), and disabled geometric augs
    # (the cache assumes a deterministic 224x224 resize and is invalidated by
    # random crops).
    use_point_supervision = config.pi3x_root is not None or config.gt_point_root is not None
    if use_point_supervision:
        flags = [
            f for f, on in (
                ("--pi3x-root", config.pi3x_root is not None),
                ("--gt-point-root", config.gt_point_root is not None),
            ) if on
        ]
        missing = [
            name
            for name, value in (
                ("--use-camvla-model", config.use_camvla_model),
                ("--use-camera-params", config.use_camera_params),
                ("--disable-geometric-augs", config.disable_geometric_augs),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{' / '.join(flags)} requires {', '.join(missing)} (cache is tied to a "
                "deterministic 224x224 resize with K-aware transforms)"
            )

    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    # Forward camera/aug flags into data configs that opt-in to them (only
    # LiberoDataConfig today). Other configs ignore unknown attributes.
    for attr in ("use_camera_params", "disable_geometric_augs"):
        if hasattr(data_config_cls, attr):
            setattr(data_config_cls, attr, getattr(config, attr))
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # Pick the dataset class. CameraAwareLeRobotDataset auto-detects camera
    # columns and falls back to the parent's behavior when they're absent, so
    # it's only swapped in when the user opts in via --use-camera-params.
    dataset_cls = CameraAwareLeRobotDataset if config.use_camera_params else LeRobotSingleDataset
    dataset_kwargs: dict = {}
    if config.use_camera_params:
        if config.pi3x_root is not None:
            dataset_kwargs["pi3x_root"] = config.pi3x_root
        if config.gt_point_root is not None:
            dataset_kwargs["gt_point_root"] = config.gt_point_root
        if config.pi3x_cam_subdirs is not None:
            dataset_kwargs["pi3x_cam_subdirs"] = config.pi3x_cam_subdirs
        # Target resolution must agree with the PointHead: 16 -> avg-pool to the
        # 16x16 patch grid; 224 -> keep full-res targets (no pooling).
        dataset_kwargs["point_target_resolution"] = config.point_head_output_resolution

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = dataset_cls(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
            **dataset_kwargs,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = dataset_cls(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                **dataset_kwargs,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Assert that the last transform is a GR00TTransform and has max_action_dim
    assert (
        hasattr(transforms, "transforms") and len(transforms.transforms) > 0
    ), "No transforms found"
    last_transform = transforms.transforms[-1]
    from gr00t.model.transforms import GR00TTransform

    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    assert hasattr(last_transform, "max_action_dim"), "GR00TTransform must have max_action_dim"
    data_max_action_dim = last_transform.max_action_dim

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
        use_camvla_model=config.use_camvla_model,  # CamVLA subclass with geometry hooks
        use_ray_embed=config.use_ray_embed,
        cross_view_type=config.cross_view_type,
        pose_enc_type=config.pose_enc_type,
        cross_view_aa_order=config.cross_view_aa_order,
        cross_view_prope_layer_idx=config.cross_view_prope_layer_idx,
        cross_view_rope_freq=config.cross_view_rope_freq,
        use_pi3x_distill=use_point_supervision,
        pi3x_loss_weight=config.pi3x_loss_weight,
        pi3x_loss_type=config.pi3x_loss_type,
        pi3x_ray_loss_weight=config.pi3x_ray_loss_weight,
        pi3x_depth_loss_weight=config.pi3x_depth_loss_weight,
        pi3x_depth_weighting=config.pi3x_depth_weighting,
        point_target_gt_weight=config.point_target_gt_weight,
        point_head_output_resolution=config.point_head_output_resolution,
        action_loss_weight=config.action_loss_weight,
        trainable_prefixes=config.trainable_prefixes,
    )

    # Update action_horizon and max_action_dim to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim

    if action_horizon_mismatch or action_dim_mismatch:
        # Store old values for logging
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            print(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

        # Update the action head config (need to copy to avoid modifying original)
        import copy

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        if not action_dim_mismatch:
            print("Copying weights from old action head (compatible dimensions)")
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            print(
                f"Partial weight copy: copying first {old_action_dim} dimensions, initializing last {data_max_action_dim - old_action_dim} dimensions randomly"
            )
            new_action_head.state_dict().update(
                _copy_partial_action_expert_weights(
                    model.action_head.state_dict(),
                    new_action_head.state_dict(),
                    old_action_dim,
                    data_max_action_dim,
                )
            )

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim

        # Update the main model's action_dim for validation (critical for validate_inputs)
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=(
            {"min_lr_rate": config.min_lr_rate}
            if config.lr_scheduler_type == "cosine_with_min_lr"
            else None
        ),
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    # Stash the optional per-group LLM LR on the args object so DualBrainTrainer
    # .create_optimizer can read it without changing the TrainRunner signature.
    # TrainingArguments is a plain (non-slotted) dataclass, so this is safe.
    training_args.llm_learning_rate = config.llm_learning_rate

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
