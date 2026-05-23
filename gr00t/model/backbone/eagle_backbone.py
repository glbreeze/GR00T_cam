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
from contextlib import contextmanager

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t
from gr00t.model.backbone.eagle_camvla_model import Eagle2_5_VLCamVLA
from gr00t.model.losses.pi3x import legacy_conf_mse_loss, pi3x_point_loss


@contextmanager
def _null_context():
    yield

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        use_camvla_model: bool = False,
        use_ray_embed: bool = False,
        cross_view_type: str = "none",
        pose_enc_type: str = "null",
        cross_view_aa_order: str = "fg",
        cross_view_prope_layer_idx: tuple[int, ...] = (),
        cross_view_rope_freq: float = 100.0,
        use_pi3x_distill: bool = False,
        pi3x_loss_type: str = "pi3x_local_pointmap",
        pi3x_ray_loss_weight: float = 1.0,
        pi3x_depth_loss_weight: float = 1.0,
        pi3x_depth_weighting: str = "pi3x_inverse",
        pi3x_legacy_conf_threshold: float = 0.1,
        pi3x_legacy_order: int = 2,
        point_target_gt_weight: float = 0.5,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            use_camvla_model: if True, use the CamVLA subclass (Eagle2_5_VLCamVLA)
                which exposes hooks for geometry modules. Identical behavior to
                the parent class until those hooks are populated.
            use_ray_embed: if True, add a per-patch ray embedding from K^-1.
                Requires use_camvla_model=True and camera intrinsics in the batch.
            cross_view_type: 'none' / 'simple' / 'standard'. Selects the cross-view
                fusion topology. Requires use_camvla_model=True when != 'none'.
            pose_enc_type: 'null' / 'prope'. When 'prope', PRoPE pose-injection
                blocks are interleaved inside the standard fusion stack.
                Requires cross_view_type='standard'.
            cross_view_aa_order: ordering of frame ('f') and global ('g') blocks
                in the standard fusion stack.
            cross_view_prope_layer_idx: indices (into the 'g' sub-sequence of
                aa_order) after which PRoPE pose-injection blocks are inserted.
                Only used when pose_enc_type='prope'.
            cross_view_rope_freq: frequency base for 2D RoPE in frame/global
                blocks and for X/Y RoPE inside PRoPE attention. <= 0 disables
                RoPE in the regular frame/global blocks (PoseInjectBlock still
                uses RoPE internally with a default of 100.0).
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        # Any geometry knob set implies the CamVLA subclass.
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

        if pi3x_loss_type not in ("pi3x_local_pointmap", "legacy_conf_mse"):
            raise ValueError(
                f"pi3x_loss_type must be 'pi3x_local_pointmap' or 'legacy_conf_mse', "
                f"got {pi3x_loss_type!r}"
            )

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        if use_camvla_model:
            self.eagle_model = Eagle2_5_VLCamVLA(config)
            self.eagle_model.build_geometry_modules(
                use_ray_embed=use_ray_embed,
                cross_view_type=cross_view_type,
                pose_enc_type=pose_enc_type,
                aa_order=cross_view_aa_order,
                prope_layer_idx=cross_view_prope_layer_idx,
                rope_freq=cross_view_rope_freq,
                use_pi3x_distill=use_pi3x_distill,
            )
        else:
            self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        self._use_camvla_model = use_camvla_model
        self._use_pi3x_distill = use_pi3x_distill
        self._pi3x_loss_type = pi3x_loss_type
        self._pi3x_ray_loss_weight = float(pi3x_ray_loss_weight)
        self._pi3x_depth_loss_weight = float(pi3x_depth_loss_weight)
        self._pi3x_depth_weighting = pi3x_depth_weighting
        self._pi3x_legacy_conf_threshold = float(pi3x_legacy_conf_threshold)
        self._pi3x_legacy_order = int(pi3x_legacy_order)
        # Dual-loss mix when both pi3x.* (teacher) and gt.* (ground truth)
        # targets are present: total = w_gt * L(pred, gt) + (1-w_gt) * L(pred, pi3x).
        # Ignored when only one channel is supplied.
        self._point_target_gt_weight = float(point_target_gt_weight)
        if not 0.0 <= self._point_target_gt_weight <= 1.0:
            raise ValueError(
                f"point_target_gt_weight must be in [0, 1], got {self._point_target_gt_weight}"
            )

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        # Route camera params (if present) to the CamVLA model's extract_feature
        # via a context manager. The HF forward signature doesn't accept them,
        # so we stash them on the model object for the duration of the call.
        camera_params = {k: v for k, v in vl_input.items() if k.startswith("camera.")}
        if self._use_camvla_model and camera_params:
            ctx = self.eagle_model.camera_params_context(camera_params)
        else:
            ctx = self.eagle_model.camera_params_context(None) if self._use_camvla_model else _null_context()

        with ctx:
            eagle_output = self.eagle_model(
                **eagle_input, output_hidden_states=True, return_dict=True
            )
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def _point_loss_for_channel(
        self, vl_input: BatchFeature, prefix: str, preds: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Run the configured point loss for one target channel (``pi3x`` or ``gt``).

        Returns ``(total, aux_xy, aux_z)`` — for ``pi3x_local_pointmap`` the
        components are (ray_loss, depth_loss); for ``legacy_conf_mse`` they are
        (xy_loss, z_loss). Returns ``None`` if the channel's targets are absent.
        """
        if not all(k in vl_input for k in (f"{prefix}.xy", f"{prefix}.logz", f"{prefix}.conf")):
            return None

        xy_pred, logz_pred = preds  # (B, V, P, 2), (B, V, P, 1)
        B, V_pred, P, _ = xy_pred.shape

        # Targets arrive as (B, V, H, W, *) — flatten the H*W axes into P to
        # match the head's output. We cast to the head's dtype (bf16 in
        # training) so loss ops don't auto-upcast.
        def _to_BVP(t: torch.Tensor, last_dim: int) -> torch.Tensor:
            assert t.ndim == 5, f"expected (B, V, H, W, {last_dim}), got shape {tuple(t.shape)}"
            return t.reshape(B, t.shape[1], -1, last_dim).to(dtype=xy_pred.dtype)

        xy_t = _to_BVP(vl_input[f"{prefix}.xy"], 2)
        logz_t = _to_BVP(vl_input[f"{prefix}.logz"], 1)
        conf_t = _to_BVP(vl_input[f"{prefix}.conf"], 1)

        # Reconcile V: padded views with no target are dropped (matches openpi's
        # loss site slicing ``pred[:, :V_tgt]``).
        V_tgt = xy_t.shape[1]
        if V_tgt > V_pred:
            raise ValueError(
                f"{prefix} target has V={V_tgt} but PointHead predicted V={V_pred}; "
                "check PI3X_CAM_SUBDIRS order vs. the model-side cam order."
            )
        xy_pred = xy_pred[:, :V_tgt]
        logz_pred = logz_pred[:, :V_tgt]

        if self._pi3x_loss_type == "pi3x_local_pointmap":
            total, ray_or_xy, depth_or_z, _scale = pi3x_point_loss(
                xy_pred=xy_pred,
                logz_pred=logz_pred,
                xy_target=xy_t,
                logz_target=logz_t,
                conf_target=conf_t,
                ray_loss_weight=self._pi3x_ray_loss_weight,
                depth_loss_weight=self._pi3x_depth_loss_weight,
                depth_weighting=self._pi3x_depth_weighting,
            )
        else:  # legacy_conf_mse
            total, ray_or_xy, depth_or_z, _scale = legacy_conf_mse_loss(
                xy_pred=xy_pred,
                logz_pred=logz_pred,
                xy_target=xy_t,
                logz_target=logz_t,
                conf_target=conf_t,
                conf_threshold=self._pi3x_legacy_conf_threshold,
                order=self._pi3x_legacy_order,
            )
        return total, ray_or_xy, depth_or_z

    def _compute_point_loss(self, vl_input: BatchFeature) -> dict | None:
        """Combine PointHead predictions (stashed by ``Eagle2_5_VLCamVLA.extract_feature``)
        with point-supervision targets into an aux-loss bundle.

        Supports two target channels — ``pi3x.*`` (teacher distillation) and
        ``gt.*`` (ground-truth pointmaps). With both present, the losses are
        mixed ``w_gt * L_gt + (1 - w_gt) * L_pi3x`` (``w_gt`` =
        ``self._point_target_gt_weight``), mirroring openpi's dual-loss mode.
        With one present, that channel's loss is used directly.

        Returns a dict ``{"total", "xy", "z", "gt"?, "teacher"?}`` (the latter
        two are detached per-channel totals, present only in dual mode), or
        ``None`` when supervision is off / no preds / no targets are in the
        batch — caller treats that as "no aux loss".
        """
        if not self._use_pi3x_distill:
            return None
        preds = getattr(self.eagle_model, "_pending_pi3x_preds", None)
        if preds is None:
            return None

        gt = self._point_loss_for_channel(vl_input, "gt", preds)
        teacher = self._point_loss_for_channel(vl_input, "pi3x", preds)

        if gt is not None and teacher is not None:
            w = self._point_target_gt_weight
            total = w * gt[0] + (1.0 - w) * teacher[0]
            xy = w * gt[1] + (1.0 - w) * teacher[1]
            z = w * gt[2] + (1.0 - w) * teacher[2]
            return {
                "total": total,
                "xy": xy,
                "z": z,
                "gt": gt[0].detach(),
                "teacher": teacher[0].detach(),
            }
        single = gt if gt is not None else teacher
        if single is None:
            return None
        return {"total": single[0], "xy": single[1], "z": single[2]}

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input) # [bs, 564, 2048]

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        out = {"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}

        # Compute point-head supervision loss (if configured + targets present).
        # The caller (GR00T_N1_5.forward) weights and adds it to the action loss.
        # We also surface the (xy/ray, z/depth) components for logging, plus the
        # per-channel gt/teacher totals when both are supervised (dual-loss).
        aux = self._compute_point_loss(vl_input)
        if aux is not None:
            out["aux_loss_pi3x"] = aux["total"]
            out["aux_loss_pi3x_xy"] = aux["xy"].detach()
            out["aux_loss_pi3x_z"] = aux["z"].detach()
            if "gt" in aux:
                out["aux_loss_pi3x_gt"] = aux["gt"]
                out["aux_loss_pi3x_teacher"] = aux["teacher"]

        return BatchFeature(data=out)  # [B, T2, hidden_size]
