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


# Maps each video key to the intrinsic key the CameraAwareLeRobotDataset emits.
# Kept here (and not inside the camera-aware transforms) so it's obvious where
# the pairing convention is defined.
_LIBERO_INTRINSIC_KEYS = {
    "video.image": "camera.agent_intrinsic",
    "video.wrist_image": "camera.wrist_intrinsic",
}


class LiberoDataConfig(BaseDataConfig):
    # When True, the transform chain uses camera-aware Video* variants that
    # update agent/wrist intrinsics alongside the pixels. Pair with
    # CameraAwareLeRobotDataset, which loads the paired K columns from the
    # parquet. No-op if the dataset doesn't carry them.
    use_camera_params: bool = False

    # When True, drops the random crop from the pipeline. Used for pi3x
    # distillation: pi3x targets are cached against a deterministic 224x224
    # resize, so any per-sample geometric jitter (crop, rotation) would
    # invalidate them. Color jitter (photometric) and the resize itself stay.
    disable_geometric_augs: bool = False

    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
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
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "mean_std",
                    "action.y": "mean_std",
                    "action.z": "mean_std",
                    "action.roll": "mean_std",
                    "action.pitch": "mean_std",
                    "action.yaw": "mean_std",
                    "action.gripper": "min_max",
                },
            )
        # Pick the crop and resize variants based on the use_camera_params flag.
        # When use_camera_params=False, behavior is byte-for-byte identical to
        # the upstream chain. When True, the camera-aware versions also update
        # paired intrinsics in the sample dict.
        if self.use_camera_params:
            resize_cls = VideoResizeCameraAware
            crop_cls = VideoCropCameraAware
            resize_kwargs = {"intrinsic_keys": _LIBERO_INTRINSIC_KEYS}
            crop_kwargs = {"intrinsic_keys": _LIBERO_INTRINSIC_KEYS}
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


class LiberoDataConfigMeanStd(LiberoDataConfig):
    """Apply mean_std normalization to actions other than gripper."""

    def transform(self) -> ModalityTransform:
        return super().transform(action_norm="mean_std")
