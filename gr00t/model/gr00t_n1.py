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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3

# Substrings used to detect whether a checkpoint already carries trained
# CamVLA geometry weights. If any param name contains one of these, we skip
# ``reset_geometry_modules`` so a stage-2 run preserves stage-1's trained
# ray_embed / cross_view_fusion / point_head.
_GEOMETRY_KEY_NEEDLES: Tuple[str, ...] = (
    "ray_embed_module",
    "cross_view_fusion",
    "point_head",
    "pose_inject_blocks",
)


def _checkpoint_has_geometry_keys(local_model_path: str) -> bool:
    """Scan the safetensors index / single file at ``local_model_path`` for
    CamVLA geometry parameter names.

    Returns True iff at least one tensor key contains a geometry needle. Falls
    back to ``False`` (i.e., assume base ckpt, reset on next call) when the
    checkpoint is in a format we don't recognise -- the worst case is the same
    behaviour as before this helper existed.
    """
    p = Path(local_model_path)
    keys: Iterable[str]

    index_json = p / "model.safetensors.index.json"
    if index_json.exists():
        with index_json.open() as f:
            keys = json.load(f).get("weight_map", {}).keys()
    else:
        single = p / "model.safetensors"
        if not single.exists():
            return False
        try:
            from safetensors import safe_open  # local import; optional dep at top level
        except ImportError:
            return False
        with safe_open(str(single), framework="pt") as f:
            keys = list(f.keys())

    return any(any(n in k for n in _GEOMETRY_KEY_NEEDLES) for k in keys)


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        # Outer coefficient for the pi3x distillation loss. ``from_pretrained``
        # overrides this when distillation is enabled.
        self.pi3x_loss_weight: float = 1.0
        # Multiplier on the action (flow-matching) loss. Stage-1 geometry warmup
        # uses a small value (e.g. 0.1) so the aux distillation loss dominates;
        # stage-2 keeps it at 1.0 for action-focused fine-tuning.
        self.action_loss_weight: float = 1.0

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

        # Pi3x distillation: fold the backbone-side aux loss into the trainer
        # loss. Surface action/aux components on the output dict so DualBrainTrainer
        # can pull them into wandb logs (HF Trainer itself only logs `loss`).
        # Total = action_loss_weight * action_loss + pi3x_loss_weight * aux_loss.
        action_only = action_head_outputs["loss"]
        action_head_outputs["action_loss"] = action_only.detach()
        # Guard the multiplication so action_loss_weight=1.0 is byte-for-byte
        # identical to the pre-stage-training graph.
        total = (
            action_only
            if self.action_loss_weight == 1.0
            else self.action_loss_weight * action_only
        )
        if "aux_loss_pi3x" in backbone_outputs:
            aux = backbone_outputs["aux_loss_pi3x"]
            total = total + self.pi3x_loss_weight * aux
            action_head_outputs["aux_loss"] = aux.detach()
            action_head_outputs["aux_xy_loss"] = backbone_outputs["aux_loss_pi3x_xy"]
            action_head_outputs["aux_z_loss"] = backbone_outputs["aux_loss_pi3x_z"]
            # Dual-loss (gt + pi3x): surface the per-channel raw totals so the
            # mix weight can be tuned from the logs.
            if "aux_loss_pi3x_gt" in backbone_outputs:
                action_head_outputs["aux_gt_loss"] = backbone_outputs["aux_loss_pi3x_gt"]
                action_head_outputs["aux_pi3x_loss"] = backbone_outputs["aux_loss_pi3x_teacher"]
        action_head_outputs["loss"] = total

        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        use_camvla_model = kwargs.pop("use_camvla_model", False)
        use_ray_embed = kwargs.pop("use_ray_embed", False)
        cross_view_type = kwargs.pop("cross_view_type", "none")
        pose_enc_type = kwargs.pop("pose_enc_type", "null")
        cross_view_aa_order = kwargs.pop("cross_view_aa_order", "fg")
        cross_view_prope_layer_idx = tuple(kwargs.pop("cross_view_prope_layer_idx", ()) or ())
        cross_view_rope_freq = float(kwargs.pop("cross_view_rope_freq", 100.0))
        use_pi3x_distill = kwargs.pop("use_pi3x_distill", False)
        pi3x_loss_weight = float(kwargs.pop("pi3x_loss_weight", 1.0))
        pi3x_loss_type = kwargs.pop("pi3x_loss_type", "pi3x_local_pointmap")
        pi3x_ray_loss_weight = float(kwargs.pop("pi3x_ray_loss_weight", 1.0))
        pi3x_depth_loss_weight = float(kwargs.pop("pi3x_depth_loss_weight", 1.0))
        pi3x_depth_weighting = kwargs.pop("pi3x_depth_weighting", "pi3x_inverse")
        point_target_gt_weight = float(kwargs.pop("point_target_gt_weight", 0.5))
        action_loss_weight = float(kwargs.pop("action_loss_weight", 1.0))
        trainable_prefixes: Tuple[str, ...] = tuple(kwargs.pop("trainable_prefixes", ()) or ())
        geometry_requested = (
            use_ray_embed
            or cross_view_type != "none"
            or pose_enc_type != "null"
            or bool(cross_view_prope_layer_idx)
            or use_pi3x_distill
        )
        if geometry_requested and not use_camvla_model:
            raise ValueError(
                "use_ray_embed / cross_view_type / pose_enc_type / cross_view_prope_layer_idx / "
                "use_pi3x_distill require use_camvla_model=True"
            )

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Use CamVLA backbone subclass: {use_camvla_model}")
        print(
            f"Geometry: ray_embed={use_ray_embed} cross_view={cross_view_type} "
            f"pose_enc={pose_enc_type} aa_order={cross_view_aa_order!r} "
            f"prope_layer_idx={cross_view_prope_layer_idx} "
            f"rope_freq={cross_view_rope_freq}"
        )
        print(
            f"Pi3x distill: enabled={use_pi3x_distill} loss_type={pi3x_loss_type!r} "
            f"weight={pi3x_loss_weight} ray_w={pi3x_ray_loss_weight} "
            f"depth_w={pi3x_depth_loss_weight} depth_weighting={pi3x_depth_weighting!r} "
            f"gt_weight={point_target_gt_weight} (dual-loss mix when both gt.* and pi3x.* present)"
        )
        print(
            f"Stage weights: action_loss_weight={action_loss_weight} "
            f"trainable_prefixes={list(trainable_prefixes)}"
        )

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # Inject CamVLA flags into backbone_cfg before HF instantiates the model.
        # backbone_cfg is a nested dict in the saved config; passing it via **kwargs
        # would clobber the whole dict, so load the config explicitly and mutate it.
        if use_camvla_model:
            config = GR00T_N1_5_Config.from_pretrained(local_model_path)
            config.backbone_cfg["use_camvla_model"] = True
            config.backbone_cfg["use_ray_embed"] = use_ray_embed
            config.backbone_cfg["cross_view_type"] = cross_view_type
            config.backbone_cfg["pose_enc_type"] = pose_enc_type
            config.backbone_cfg["cross_view_aa_order"] = cross_view_aa_order
            config.backbone_cfg["cross_view_prope_layer_idx"] = cross_view_prope_layer_idx
            config.backbone_cfg["cross_view_rope_freq"] = cross_view_rope_freq
            config.backbone_cfg["use_pi3x_distill"] = use_pi3x_distill
            config.backbone_cfg["pi3x_loss_type"] = pi3x_loss_type
            config.backbone_cfg["pi3x_ray_loss_weight"] = pi3x_ray_loss_weight
            config.backbone_cfg["pi3x_depth_loss_weight"] = pi3x_depth_loss_weight
            config.backbone_cfg["pi3x_depth_weighting"] = pi3x_depth_weighting
            config.backbone_cfg["point_target_gt_weight"] = point_target_gt_weight
            kwargs["config"] = config

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        # HF's _init_weights overrides our zero-init for geometry modules that
        # aren't in the checkpoint. Re-zero them so each enabled module starts
        # as identity (otherwise their large random init produces NaN in bf16).
        # Skip the reset when the checkpoint already carries trained geometry
        # weights (stage-2 loading from a stage-1 ckpt) -- otherwise we'd zero
        # out the very weights we just loaded.
        if use_camvla_model:
            if _checkpoint_has_geometry_keys(local_model_path):
                print(
                    "Geometry modules present in checkpoint -- preserving loaded "
                    "weights (skipping reset_geometry_modules)."
                )
            else:
                print(
                    "Geometry modules NOT in checkpoint -- resetting to zero/identity init."
                )
                pretrained_model.backbone.eagle_model.reset_geometry_modules()

        pretrained_model.pi3x_loss_weight = pi3x_loss_weight
        pretrained_model.action_loss_weight = action_loss_weight

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        # Stage-1 geometry warmup: freeze everything outside the listed prefixes.
        # Applied AFTER the per-module set_trainable_parameters calls above so it
        # is the final word on requires_grad. Matches openpi's ``trainable_prefixes``.
        if trainable_prefixes:
            matched = 0
            total = 0
            for name, p in pretrained_model.named_parameters():
                total += 1
                if any(name.startswith(prefix) for prefix in trainable_prefixes):
                    p.requires_grad = True
                    matched += 1
                else:
                    p.requires_grad = False
            print(
                f"trainable_prefixes={list(trainable_prefixes)} -> "
                f"{matched}/{total} params trainable"
            )
            if matched == 0:
                raise ValueError(
                    f"trainable_prefixes={list(trainable_prefixes)} matched 0 parameters. "
                    "Check the prefix against model.named_parameters() (top-level keys "
                    "are 'backbone.eagle_model.*' and 'action_head.*')."
                )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
