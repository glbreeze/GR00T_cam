"""Camera-aware video transforms that update paired intrinsics.

These subclasses extend `VideoResize` and `VideoCrop` to also update camera
intrinsics so that K stays consistent with the transformed pixel grid.
Extrinsics are not touched by 2D image transforms (camera pose in the world
doesn't change when the image is resized or cropped).

Each transform is given an `intrinsic_keys` mapping that pairs a video key
with the intrinsic key it updates, e.g.::

    VideoResizeCameraAware(
        apply_to=["video.image", "video.wrist_image"],
        height=224, width=224,
        intrinsic_keys={
            "video.image": "camera.agent_intrinsic",
            "video.wrist_image": "camera.wrist_intrinsic",
        },
    )

If a paired intrinsic key is missing from the sample dict, the K update is
silently skipped — that makes the transform safe to use in pipelines where
camera params aren't loaded (e.g., when running without
`CameraAwareLeRobotDataset`).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torchvision.transforms.v2 as T
from pydantic import Field

from gr00t.data.transform.video import VideoCrop, VideoResize


def _apply_pixel_transform_to_K(K: np.ndarray | torch.Tensor, pixel_T: np.ndarray) -> np.ndarray | torch.Tensor:
    """Apply a 3x3 pixel transform T to one or many intrinsics: K' = T @ K.

    K is shape [..., 3, 3]. Works for numpy arrays and torch tensors.
    """
    if isinstance(K, torch.Tensor):
        T_t = torch.as_tensor(pixel_T, dtype=K.dtype, device=K.device)
        return T_t @ K
    return pixel_T.astype(K.dtype) @ K


def _resize_pixel_transform(h_in: int, w_in: int, h_out: int, w_out: int) -> np.ndarray:
    """3x3 pixel transform for a pure resize (no padding, no crop)."""
    sx = w_out / w_in
    sy = h_out / h_in
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _crop_pixel_transform(start_h: int, start_w: int) -> np.ndarray:
    """3x3 pixel transform for a crop that translates the origin to (start_h, start_w).

    Crop alone scales nothing — it just shifts cx, cy. (If a resize-after-crop
    is added later, its scale matrix composes on the left.)
    """
    return np.array([[1.0, 0.0, -start_w], [0.0, 1.0, -start_h], [0.0, 0.0, 1.0]], dtype=np.float64)


class VideoResizeCameraAware(VideoResize):
    """`VideoResize` + paired intrinsic update."""

    intrinsic_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Maps each video key in `apply_to` to its paired intrinsic key in the sample dict.",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        # Snapshot input dims per video key BEFORE resize so we can compute the scale matrix.
        input_dims: dict[str, tuple[int, int]] = {}
        for vk in self.apply_to:
            v = data[vk]
            if self.backend == "torchvision":
                # Tensor layout: (T, C, H, W) or (T, B, C, H, W)
                input_dims[vk] = (v.shape[-2], v.shape[-1])
            elif self.backend == "albumentations":
                # Array layout: (T, H, W, C) or (T, B, H, W, C)
                input_dims[vk] = (v.shape[-3], v.shape[-2])
            else:
                raise ValueError(f"Backend {self.backend} not supported")

        data = super().apply(data)

        for vk, ik in self.intrinsic_keys.items():
            if ik not in data:
                continue
            h_in, w_in = input_dims[vk]
            pixel_T = _resize_pixel_transform(h_in, w_in, self.height, self.width)
            data[ik] = _apply_pixel_transform_to_K(data[ik], pixel_T)

        return data


class VideoCropCameraAware(VideoCrop):
    """`VideoCrop` + paired intrinsic update.

    Samples a single crop offset per call and applies it to every video in
    `apply_to` so all views remain spatially aligned (consistent with the
    parent). The matching intrinsic updates are then applied per-view.

    Only the torchvision backend is implemented — albumentations adds Replay
    machinery that makes capturing the sampled offset awkward, and we use the
    torchvision backend by default in the Libero config.
    """

    intrinsic_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Maps each video key in `apply_to` to its paired intrinsic key in the sample dict.",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.backend != "torchvision":
            raise NotImplementedError(
                "VideoCropCameraAware only supports backend='torchvision' "
                f"(got {self.backend!r}). Use VideoCrop instead if you need albumentations."
            )

        # self.height / self.width are set to the input resolution in
        # VideoCrop.get_transform(); crop size scales those by self.scale.
        crop_h = int(self.height * self.scale)
        crop_w = int(self.width * self.scale)

        # Validate input shapes match the configured original resolution. This
        # also matches what VideoCrop.check_input enforces.
        for vk in self.apply_to:
            v = data[vk]
            h, w = v.shape[-2], v.shape[-1]
            assert (h, w) == (self.height, self.width), (
                f"Video {vk} has shape ({h}, {w}), expected ({self.height}, {self.width})"
            )

        if self.training:
            max_h = self.height - crop_h
            max_w = self.width - crop_w
            start_h = int(torch.randint(0, max_h + 1, (1,)).item()) if max_h > 0 else 0
            start_w = int(torch.randint(0, max_w + 1, (1,)).item()) if max_w > 0 else 0
        else:
            # Center crop in eval mode (matches parent behavior).
            start_h = (self.height - crop_h) // 2
            start_w = (self.width - crop_w) // 2

        # Apply the same crop to every view.
        for vk in self.apply_to:
            data[vk] = T.functional.crop(
                data[vk], top=start_h, left=start_w, height=crop_h, width=crop_w
            )

        # Update paired intrinsics.
        if self.intrinsic_keys:
            pixel_T = _crop_pixel_transform(start_h, start_w)
            for vk, ik in self.intrinsic_keys.items():
                if ik not in data:
                    continue
                data[ik] = _apply_pixel_transform_to_K(data[ik], pixel_T)

        return data
