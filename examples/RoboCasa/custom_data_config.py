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

"""GR00T N1.5 data config for the RoboCasa v0 / human50 24-task bundle.

Mirrors examples/Libero/custom_data_config.py but for RoboCasa's mobile-base
single-arm layout (16-dim state, 12-dim action, two cameras). The on-disk
GR00T-flavor dataset is produced by convert_openpi_lerobot_to_gr00t.py from the
openpi-flavor `robocasa24/all24_human_camaware` LeRobot dataset. The intended
counterpart in openpi is `pi0_robocasa24_all24_baseline` (and the
`..._cam_prope_ray_view_distill_fullres_stage{1,2}` configs for the camera-aware
path, which this config enables via the use_camera_params flag).
"""

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.data.transform.video_camvla import VideoCropCameraAware, VideoResizeCameraAware
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


# Maps each video key to the intrinsic key CameraAwareLeRobotDataset emits. The
# converter renames RoboCasa's agentview_left/eye_in_hand camera columns to the
# agent_/wrist_ convention the dataset auto-detects, so agent==image (front cam),
# wrist==wrist_image (eye-in-hand). Kept here so the pairing convention is
# obvious (mirrors examples/Libero/custom_data_config.py::_LIBERO_INTRINSIC_KEYS).
_ROBOCASA_INTRINSIC_KEYS = {
    "video.image": "camera.agent_intrinsic",
    "video.wrist_image": "camera.wrist_intrinsic",
}


class RobocasaDataConfig(BaseDataConfig):
    # When True, the transform chain uses camera-aware Video* variants that
    # update agent/wrist intrinsics alongside the pixels. Pair with
    # CameraAwareLeRobotDataset, which loads the paired K columns from the
    # parquet. No-op if the dataset doesn't carry them.
    use_camera_params: bool = False

    # When True, drops the random crop from the pipeline. Used for pi3x / GT
    # point-target supervision: targets are cached against a deterministic
    # 224x224 resize, so any per-sample geometric jitter (crop, rotation) would
    # invalidate them. Color jitter (photometric) and the resize itself stay.
    disable_geometric_augs: bool = False

    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    # RoboCasa observation.state is a flat 16-dim vector; the slices below match
    # examples/RoboCasa/modality.json (base pose + base->eef pose + gripper qpos).
    state_keys = [
        "state.base_pos",
        "state.base_quat",
        "state.base_to_eef_pos",
        "state.base_to_eef_quat",
        "state.gripper_qpos",
    ]
    # RoboCasa action is a flat 12-dim vector (absolute eef targets, not deltas).
    # Only control_mode (dim 4) is truly constant in the human demos -> it has
    # zero range and min_max maps it to 0 (StateActionTransform sets min==max
    # dims to 0). base_motion (dims 0:4) is *not* constant -- it has small but
    # nonzero ranges, so min_max rescales those small base motions to [-1, 1].
    action_keys = [
        "action.base_motion",
        "action.control_mode",
        "action.eef_pos",
        "action.eef_rot",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:
        if action_norm == "min_max":
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            )
        else:
            # mean_std for the continuous arm/base channels, min_max for the
            # gripper (bounded, bang-bang). Mirrors the Libero mean_std variant.
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.base_motion": "mean_std",
                    "action.control_mode": "mean_std",
                    "action.eef_pos": "mean_std",
                    "action.eef_rot": "mean_std",
                    "action.gripper": "min_max",
                },
            )
        # Pick the crop and resize variants based on use_camera_params. When
        # False, behavior is byte-for-byte identical to the upstream chain. When
        # True, the camera-aware versions also update paired intrinsics.
        if self.use_camera_params:
            resize_cls = VideoResizeCameraAware
            crop_cls = VideoCropCameraAware
            resize_kwargs = {"intrinsic_keys": _ROBOCASA_INTRINSIC_KEYS}
            crop_kwargs = {"intrinsic_keys": _ROBOCASA_INTRINSIC_KEYS}
        else:
            resize_cls = VideoResize
            crop_cls = VideoCrop
            resize_kwargs = {}
            crop_kwargs = {}

        video_transforms = [VideoToTensor(apply_to=self.video_keys)]
        if not self.disable_geometric_augs:
            video_transforms.append(
                crop_cls(apply_to=self.video_keys, scale=0.95, **crop_kwargs)
            )
        video_transforms.append(
            resize_cls(
                apply_to=self.video_keys,
                height=224,
                width=224,
                interpolation="linear",
                **resize_kwargs,
            )
        )
        video_transforms.extend(
            [
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                VideoToNumpy(apply_to=self.video_keys),
            ]
        )

        transforms = [
            # video transforms
            *video_transforms,
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            action_transform,
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RobocasaDataConfigMeanStd(RobocasaDataConfig):
    """Apply mean_std normalization to actions other than gripper."""

    def transform(self) -> ModalityTransform:
        return super().transform(action_norm="mean_std")
