"""Camera-aware LeRobot dataset subclass for the CamVLA project.

Extends :class:`LeRobotSingleDataset` to load camera intrinsic / extrinsic
columns directly from the parquet, bypassing ``modality.json`` (which only
supports 1D state slices, while K is (3,3) and [R|t] is (4,4)).

Camera columns are auto-detected: if a dataset's parquet schema does not
contain any of the expected columns, this class behaves identically to the
parent. That makes it safe to drop in as the default loader for any dataset.

When ``pi3x_root`` (teacher distillation) or ``gt_point_root`` (ground-truth
pointmaps from the simulator) is provided, the class also loads per-frame
point-supervision targets from ``{root}/{cam_subdir}/episode_{NNNNNN}.npz``
(one file per camera, indexed by ``trajectory_id`` and ``base_index``). The
two sources share the same on-disk schema (``xy`` / ``log_z`` / ``conf``) and
the same PointHead loss path, so only one may be set at a time. Targets are
emitted as ``data["pi3x.{xy,logz,conf}"]`` of shape ``(V, H, W, *)`` where
``V`` is the number of cameras with cached targets (silently skipped if any
cam's npz is missing).
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

from gr00t.data.dataset import LeRobotSingleDataset


class CameraAwareLeRobotDataset(LeRobotSingleDataset):

    # Columns are read verbatim from the parquet (no rescaling here — pixel
    # transforms from image augs are applied later by camera-aware video
    # transforms). Subclass and override to support different column names.
    CAMERA_COLUMNS: tuple[str, ...] = (
        "agent_intrinsic",
        "agent_extrinsic",
        "wrist_intrinsic",
        "wrist_extrinsic",
    )

    # Order matters — entries are stacked along the V (view) axis. Must match
    # the per-view image order the CamVLA backbone sees (agent before wrist).
    # Override in a subclass to support different cameras.
    PI3X_CAM_SUBDIRS: tuple[str, ...] = ("agent", "wrist")

    # Target grid the PointHead expects (16x16 patches at the SigLIP grid).
    # If the cache is at full image resolution (e.g., 224x224), each (H, W)
    # block of this size is pooled into one target patch.
    PI3X_TARGET_PATCH: tuple[int, int] = (16, 16)

    # The key under which the dataset's video modality lives in modality_configs.
    # We reuse its delta_indices so camera params line up time-wise with the
    # frames the model sees.
    _VIDEO_MODALITY_NAME: str = "video"

    def __init__(
        self,
        *args,
        pi3x_root: str | None = None,
        gt_point_root: str | None = None,
        pi3x_cam_subdirs: tuple[str, ...] | None = None,
        point_target_resolution: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if point_target_resolution not in (16, 224):
            raise ValueError(
                f"point_target_resolution must be 16 or 224, got {point_target_resolution}."
            )
        # 16 -> avg-pool the cache frame to the 16x16 patch grid (PointHead Linear
        # head); 224 -> keep the full-res frame (PointHead ConvHead upsampler).
        # Must match --point-head-output-resolution on the model side.
        self._point_target_resolution = int(point_target_resolution)
        self._available_camera_columns = self._detect_camera_columns()
        self._camera_delta_indices = self._resolve_camera_delta_indices()

        if self._available_camera_columns:
            print(
                f"[CameraAware] {self._dataset_name}: loading camera columns "
                f"{list(self._available_camera_columns)} "
                f"with delta_indices={self._camera_delta_indices.tolist()}"
            )
        else:
            print(
                f"[CameraAware] {self._dataset_name}: no camera columns detected, "
                "falling back to LeRobotSingleDataset behavior"
            )

        # Point-head supervision can draw from up to two caches. Both share the
        # same on-disk schema ({xy, log_z, conf} per cam per episode) and the
        # same PointHead loss, but enter on distinct batch channels so the loss
        # site can supervise against each independently:
        #   * pi3x_root     -> teacher (pi3x) predictions  -> emitted as pi3x.*
        #   * gt_point_root -> ground-truth pointmaps (sim) -> emitted as gt.*
        # Setting both is allowed: the model then combines them as
        # ``alpha * L(pred, gt) + (1 - alpha) * L(pred, pi3x)`` (dual-loss mode,
        # mirroring openpi's DualPointTargetLoader). The alpha lives model-side
        # (--point-target-gt-weight); the dataset just emits whichever channels
        # are configured.
        self._pi3x_cam_subdirs: tuple[str, ...] = (
            tuple(pi3x_cam_subdirs) if pi3x_cam_subdirs else self.PI3X_CAM_SUBDIRS
        )
        self._pi3x_root: pathlib.Path | None = (
            pathlib.Path(pi3x_root).expanduser() if pi3x_root else None
        )
        self._gt_root: pathlib.Path | None = (
            pathlib.Path(gt_point_root).expanduser() if gt_point_root else None
        )
        self._available_pi3x_subdirs: tuple[str, ...] = self._detect_subdirs(self._pi3x_root)
        self._available_gt_subdirs: tuple[str, ...] = self._detect_subdirs(self._gt_root)

        for label, root, subs in (
            ("pi3x", self._pi3x_root, self._available_pi3x_subdirs),
            ("gt", self._gt_root, self._available_gt_subdirs),
        ):
            if root is None:
                continue
            if subs:
                print(
                    f"[CameraAware] {self._dataset_name}: loading {label} point targets "
                    f"from {root} for views {list(subs)}"
                )
            else:
                print(
                    f"[CameraAware] {self._dataset_name}: {label} point-target root={root} "
                    f"but no expected cam subdirs {list(self._pi3x_cam_subdirs)} found — "
                    f"no {label} targets will be emitted"
                )

    def _detect_camera_columns(self) -> tuple[str, ...]:
        """Read the first parquet's schema to determine which camera columns exist."""
        first_traj_id = int(self.trajectory_ids[0])
        chunk = self.get_episode_chunk(first_traj_id)
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk, episode_index=first_traj_id
        )
        schema_names = set(pq.read_schema(parquet_path).names)
        return tuple(c for c in self.CAMERA_COLUMNS if c in schema_names)

    def _resolve_camera_delta_indices(self) -> np.ndarray:
        """Pick the delta_indices to use for camera params.

        Camera params are an observation, not an action, so they should follow
        the video time axis. We borrow the first registered video key's delta
        indices. If no video keys are configured (unusual), fall back to [0].
        """
        video_keys = self._modality_keys.get(self._VIDEO_MODALITY_NAME, [])
        if video_keys:
            return self._delta_indices[video_keys[0]]
        return np.array([0])

    @property
    def has_camera_params(self) -> bool:
        return bool(self._available_camera_columns)

    @property
    def available_camera_columns(self) -> tuple[str, ...]:
        return self._available_camera_columns

    @property
    def has_pi3x_targets(self) -> bool:
        return bool(self._available_pi3x_subdirs)

    @property
    def has_gt_targets(self) -> bool:
        return bool(self._available_gt_subdirs)

    def _detect_subdirs(self, root: pathlib.Path | None) -> tuple[str, ...]:
        """Check which configured cam subdirs actually exist on disk under ``root``.

        Auto-detection mirrors the camera-columns behavior: if the root is
        missing or a cam is absent, that cam is silently skipped. We don't
        verify per-episode files here — those are opened lazily in
        :meth:`get_step_data` and a missing-file error will surface there.
        """
        if root is None or not root.exists():
            return ()
        return tuple(sub for sub in self._pi3x_cam_subdirs if (root / sub).is_dir())

    @staticmethod
    def _avg_pool_to_patch(arr: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
        """Average-pool a ``(V, H, W, C)`` array down to ``(V, patch_h, patch_w, C)``.

        Matches openpi's ``cache_pi3x_targets.py`` 14x14 pooling when
        ``H = W = 224`` and the patch is ``16``. No-op when already at the
        target resolution. Pooling kernels must divide H and W evenly.
        """
        V, H, W, C = arr.shape
        if (H, W) == (patch_h, patch_w):
            return arr
        if H % patch_h or W % patch_w:
            raise ValueError(
                f"pi3x target spatial shape ({H}, {W}) is not divisible by "
                f"patch grid ({patch_h}, {patch_w})"
            )
        kh, kw = H // patch_h, W // patch_w
        # (V, patch_h, kh, patch_w, kw, C) -> mean over (kh, kw)
        return arr.reshape(V, patch_h, kh, patch_w, kw, C).mean(axis=(2, 4))

    @staticmethod
    def _load_cam_frame(
        cam_dir: pathlib.Path, trajectory_id: int, base_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load one frame's (xy, log_z, conf) for a single cam.

        Two on-disk layouts are supported, preferred in this order:

        * **Uncompressed per-array ``.npy``** at
          ``{cam_dir}/episode_{NNNNNN}/{xy,log_z,conf}.npy``. Opened with
          ``mmap_mode='r'``, so indexing ``[base_index]`` touches only that
          frame's pages — ~one frame of I/O, no decompression.
        * **Legacy per-episode ``.npz``** at
          ``{cam_dir}/episode_{NNNNNN}.npz``. ``mmap_mode`` is *ignored* for
          (compressed) npz, so this fully decompresses the whole episode per
          frame — kept only for backward compatibility; prefer the .npy cache.
        """
        ep = f"episode_{int(trajectory_id):06d}"
        npy_dir = cam_dir / ep
        if npy_dir.is_dir():
            def _one(name: str) -> np.ndarray:
                arr = np.load(npy_dir / f"{name}.npy", mmap_mode="r")
                return np.asarray(arr[base_index], dtype=np.float32)
            return _one("xy"), _one("log_z"), _one("conf")
        with np.load(cam_dir / f"{ep}.npz", mmap_mode="r") as f:
            return (
                np.asarray(f["xy"][base_index], dtype=np.float32),
                np.asarray(f["log_z"][base_index], dtype=np.float32),
                np.asarray(f["conf"][base_index], dtype=np.float32),
            )

    def _load_point_targets(
        self,
        root: pathlib.Path | None,
        subdirs: tuple[str, ...],
        prefix: str,
        trajectory_id: int,
        base_index: int,
    ) -> dict[str, np.ndarray] | None:
        """Load one frame of (xy, log_z, conf) for each available cam under ``root``.

        Returns ``None`` if no cams are available. Otherwise emits three arrays
        of shape ``(V, R, R, *)`` stacked over ``subdirs``, keyed
        ``{prefix}.{xy,logz,conf}`` (``prefix`` is ``"pi3x"`` for the teacher
        cache, ``"gt"`` for the ground-truth cache).

        ``R`` is set by ``point_target_resolution``: when 16, each full-res frame
        is avg-pooled to the 16x16 patch grid (matching openpi's
        ``output_resolution=16``); when 224, the full-res frame is kept as-is
        (matching openpi's ``output_resolution=224``). The pooling/keep choice
        must match the PointHead's output resolution.
        """
        if not subdirs:
            return None

        xy_views, logz_views, conf_views = [], [], []
        for sub in subdirs:
            xy, logz, conf = self._load_cam_frame(root / sub, trajectory_id, base_index)
            xy_views.append(xy)
            logz_views.append(logz)
            conf_views.append(conf)

        xy = np.stack(xy_views, axis=0)      # (V, H, W, 2)
        logz = np.stack(logz_views, axis=0)  # (V, H, W, 1)
        conf = np.stack(conf_views, axis=0)  # (V, H, W, 1)

        if self._point_target_resolution == 16:
            patch_h, patch_w = self.PI3X_TARGET_PATCH
            xy = self._avg_pool_to_patch(xy, patch_h, patch_w)
            logz = self._avg_pool_to_patch(logz, patch_h, patch_w)
            conf = self._avg_pool_to_patch(conf, patch_h, patch_w)
        else:
            R = self._point_target_resolution
            H, W = xy.shape[1], xy.shape[2]
            if (H, W) != (R, R):
                raise ValueError(
                    f"point_target_resolution={R} but {prefix} cache frame is "
                    f"({H}, {W}); full-res mode needs a cache at that resolution."
                )

        return {
            f"{prefix}.xy": xy,
            f"{prefix}.logz": logz,
            f"{prefix}.conf": conf,
        }

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)

        if self._available_camera_columns:
            traj = self.curr_traj_data  # set by super().get_step_data() via get_trajectory_data
            traj_len = len(traj)
            step_indices = base_index + self._camera_delta_indices
            # Clamp out-of-range indices to first/last frame — matches how the
            # parent handles out-of-range observation indices in retrieve_data_and_pad.
            step_indices = np.clip(step_indices, 0, traj_len - 1)

            for col in self._available_camera_columns:
                cells: Iterable = traj[col].iloc[step_indices]
                # Each cell is a length-N object-dtype np.ndarray whose elements are
                # 1D float arrays (one per row of a (3,3) or (4,4) matrix). Going
                # through .tolist() unwraps them into nested Python lists that
                # np.array can vstack into a proper [T, *col_shape] float32 tensor.
                arr = np.array([np.asarray(c).tolist() for c in cells], dtype=np.float32)
                data[f"camera.{col}"] = arr

        # Emit whichever point-target channels are configured. With both set,
        # the model supervises against each (dual-loss); see eagle_backbone.
        for root, subdirs, prefix in (
            (self._pi3x_root, self._available_pi3x_subdirs, "pi3x"),
            (self._gt_root, self._available_gt_subdirs, "gt"),
        ):
            targets = self._load_point_targets(root, subdirs, prefix, trajectory_id, base_index)
            if targets is not None:
                data.update(targets)

        return data
