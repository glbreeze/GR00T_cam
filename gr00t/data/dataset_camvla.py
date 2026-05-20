"""Camera-aware LeRobot dataset subclass for the CamVLA project.

Extends :class:`LeRobotSingleDataset` to load camera intrinsic / extrinsic
columns directly from the parquet, bypassing ``modality.json`` (which only
supports 1D state slices, while K is (3,3) and [R|t] is (4,4)).

Camera columns are auto-detected: if a dataset's parquet schema does not
contain any of the expected columns, this class behaves identically to the
parent. That makes it safe to drop in as the default loader for any dataset.
"""

from __future__ import annotations

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

    # The key under which the dataset's video modality lives in modality_configs.
    # We reuse its delta_indices so camera params line up time-wise with the
    # frames the model sees.
    _VIDEO_MODALITY_NAME: str = "video"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        if not self._available_camera_columns:
            return data

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

        return data
