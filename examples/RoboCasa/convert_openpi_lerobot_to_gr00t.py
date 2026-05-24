"""Convert openpi-flavor RoboCasa LeRobot (PNG image dtype) to GR00T-flavor (MP4 video dtype).

Source: the `robocasa24/all24_human_camaware` LeRobot v2.1 dataset (as consumed by
openpi's `pi0_robocasa24_all24_baseline`). Columns:
    observation.images.robot0_agentview_left   (image dtype, front cam)
    observation.images.robot0_eye_in_hand       (image dtype, wrist cam)
    observation.state                            (16-dim) , action (12-dim)
    observation.<cam>_extrinsic / _intrinsic     <cam> in {agentview_left, eye_in_hand}
    timestamp, frame_index, episode_index, index, task_index

Target (GR00T LeRobot, per getting_started/LeRobot_compatible_data_schema.md):
    data/chunk-NNN/episode_NNNNNN.parquet
        # image cols dropped; observation.state/action kept; camera columns renamed to
        #   agent_/wrist_ {intrinsic,extrinsic} (the convention CameraAwareLeRobotDataset
        #   auto-detects); add annotation.human.action.task_description, next.reward, next.done.
    videos/chunk-NNN/observation.images.image/episode_NNNNNN.mp4        (<- robot0_agentview_left)
    videos/chunk-NNN/observation.images.wrist_image/episode_NNNNNN.mp4  (<- robot0_eye_in_hand)
    meta/info.json        # rewritten: image features -> video features, camera cols renamed
    meta/modality.json    # copied from examples/RoboCasa/modality.json
    meta/{episodes,episodes_stats,tasks}.jsonl, stats.json

Differs from the Libero converter (examples/Libero/) only in the column maps below:
RoboCasa's video columns are namespaced (observation.images.robot0_*), its state/action
are already named observation.state/action (no rename), and its camera columns use the
agentview_left/eye_in_hand naming that we remap to agent_/wrist_.
"""

from __future__ import annotations

import io
import json
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
from PIL import Image

# Source image column -> output video suffix (becomes observation.images.<suffix> and
# the videos/.../observation.images.<suffix>/ dir). agent==front, wrist==eye-in-hand.
VIDEO_KEY_MAP = {
    "observation.images.robot0_agentview_left": "image",
    "observation.images.robot0_eye_in_hand": "wrist_image",
}
DROP_COLUMNS = set(VIDEO_KEY_MAP)

# Rename RoboCasa's per-cam K/T columns to the agent_/wrist_ convention that
# CameraAwareLeRobotDataset auto-detects (see gr00t/data/dataset_camvla.py).
RENAME_MAP = {
    "observation.agentview_left_intrinsic": "agent_intrinsic",
    "observation.agentview_left_extrinsic": "agent_extrinsic",
    "observation.eye_in_hand_intrinsic": "wrist_intrinsic",
    "observation.eye_in_hand_extrinsic": "wrist_extrinsic",
}

# Added to every row to satisfy GR00T's schema. task_index already maps to the
# language string in tasks.jsonl, so the annotation column is just an alias.
NEW_COLUMNS = ("annotation.human.action.task_description", "next.reward", "next.done")


def _modality_json_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "modality.json"


def _decode_image(cell: dict[str, Any]) -> np.ndarray:
    """Decode a HuggingFace `Image` feature cell (PNG bytes dict) to uint8 HxWxC."""
    if not isinstance(cell, dict) or "bytes" not in cell:
        raise TypeError(f"Expected image dict with 'bytes' key, got: {type(cell)!r}")
    img = Image.open(io.BytesIO(cell["bytes"]))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _encode_episode_video(
    frames: list[np.ndarray],
    out_path: pathlib.Path,
    fps: int,
    codec: str = "h264",
    pix_fmt: str = "yuv420p",
    crf: int = 23,
) -> None:
    """Encode a list of HxWxC uint8 RGB frames to MP4 (H.264 yuv420p) via pyav."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = frames[0].shape
    if height % 2 or width % 2:
        raise ValueError(
            f"H.264 yuv420p requires even dimensions, got {height}x{width} for {out_path}"
        )

    container = av.open(str(out_path), mode="w")
    try:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt
        stream.options = {"crf": str(crf)}

        for arr in frames:
            video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _rewrite_parquet(src_table: pa.Table, episode_length: int) -> pa.Table:
    """Drop image cols, rename camera cols, add annotation/next.* columns."""
    cols: dict[str, pa.Array] = {}
    for name in src_table.column_names:
        if name in DROP_COLUMNS:
            continue
        dst_name = RENAME_MAP.get(name, name)
        cols[dst_name] = src_table[name]

    if "task_index" not in cols:
        raise ValueError("Source parquet missing task_index column")
    cols["annotation.human.action.task_description"] = cols["task_index"]

    # next.reward = 0.0; next.done = True only on the last frame.
    cols["next.reward"] = pa.array(np.zeros(episode_length, dtype=np.float64))
    done = np.zeros(episode_length, dtype=bool)
    if episode_length > 0:
        done[-1] = True
    cols["next.done"] = pa.array(done)

    return pa.table(cols)


def _process_episode(
    src_parquet: pathlib.Path,
    chunk_dir_rel: str,
    out_root: pathlib.Path,
    fps: int,
    crf: int,
) -> tuple[int, int]:
    """Convert one episode. Returns (episode_index, frame_count)."""
    table = pq.read_table(src_parquet)
    n = table.num_rows
    episode_index = int(table["episode_index"][0].as_py())

    # 1. Encode each camera to its own MP4 under videos/<chunk>/observation.images.<suffix>/.
    for src_key, suffix in VIDEO_KEY_MAP.items():
        if src_key not in table.column_names:
            raise ValueError(f"{src_parquet}: missing column {src_key!r}")
        frames = [_decode_image(table[src_key][i].as_py()) for i in range(n)]
        out_video = (
            out_root
            / "videos"
            / chunk_dir_rel
            / f"observation.images.{suffix}"
            / f"episode_{episode_index:06d}.mp4"
        )
        _encode_episode_video(frames, out_video, fps=fps, crf=crf)

    # 2. Rewrite parquet (drop image cols, rename camera cols, add annotation/next.*).
    new_table = _rewrite_parquet(table, n)
    out_parquet = out_root / "data" / chunk_dir_rel / src_parquet.name
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, out_parquet)

    return episode_index, n


def _build_target_info(src_info: dict[str, Any], image_size: tuple[int, int, int]) -> dict[str, Any]:
    """Rewrite info.json: image features -> video features, rename camera cols, add new cols."""
    features = dict(src_info["features"])  # shallow copy

    # Drop image features, add video features under observation.images.<suffix>.
    for src_key, suffix in VIDEO_KEY_MAP.items():
        features.pop(src_key, None)
        features[f"observation.images.{suffix}"] = {
            "dtype": "video",
            "shape": list(image_size),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(src_info["fps"]),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    # Rename camera K/T feature keys to agent_/wrist_.
    for src_name, dst_name in RENAME_MAP.items():
        if src_name in features:
            features[dst_name] = features.pop(src_name)

    # Add the three new columns materialized in every parquet.
    features["annotation.human.action.task_description"] = {"dtype": "int64", "shape": [1]}
    features["next.reward"] = {"dtype": "float64", "shape": [1]}
    features["next.done"] = {"dtype": "bool", "shape": [1]}

    n_episodes = int(src_info["total_episodes"])
    out_info = dict(src_info)
    out_info["features"] = features
    out_info["total_videos"] = n_episodes * len(VIDEO_KEY_MAP)
    return out_info


def _compute_stats(dst_root: pathlib.Path, columns: tuple[str, ...] = ("observation.state", "action")) -> dict:
    """Compute mean/std/min/max/q01/q99 over the listed 1D columns across all parquets.

    Writing this to meta/stats.json prevents GR00T's auto-stats path from running, which
    can't handle multi-dim columns (our extrinsics are (4,4)).
    """
    buffers: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for parquet in sorted((dst_root / "data").rglob("episode_*.parquet")):
        table = pq.read_table(parquet, columns=list(columns))
        for c in columns:
            buffers[c].append(np.stack(table[c].to_numpy()).astype(np.float32))

    stats: dict[str, dict[str, list[float]]] = {}
    for c, chunks in buffers.items():
        arr = np.concatenate(chunks, axis=0)
        stats[c] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


def _copy_meta_files(
    src_meta: pathlib.Path,
    dst_meta: pathlib.Path,
    kept_episode_indices: set[int] | None = None,
) -> None:
    """Copy meta files. If `kept_episode_indices` is provided, filter episodes.jsonl
    and episodes_stats.jsonl to only those indices (needed when --max-episodes is set,
    otherwise the loader believes episodes we didn't write are available)."""
    dst_meta.mkdir(parents=True, exist_ok=True)

    def _maybe_filter_jsonl(src: pathlib.Path, dst: pathlib.Path) -> None:
        if not src.exists():
            return
        if kept_episode_indices is None:
            shutil.copy2(src, dst)
            return
        with src.open() as fin, dst.open("w") as fout:
            for line in fin:
                row = json.loads(line)
                if row.get("episode_index") in kept_episode_indices:
                    fout.write(line)

    _maybe_filter_jsonl(src_meta / "episodes.jsonl", dst_meta / "episodes.jsonl")
    _maybe_filter_jsonl(src_meta / "episodes_stats.jsonl", dst_meta / "episodes_stats.jsonl")
    # tasks.jsonl is keyed by task_index, not episode_index — always copy whole.
    if (src_meta / "tasks.jsonl").exists():
        shutil.copy2(src_meta / "tasks.jsonl", dst_meta / "tasks.jsonl")


def _iter_chunked_episodes(src_root: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    """Yield (parquet_path, chunk_dir_relative) for every episode parquet."""
    out: list[tuple[pathlib.Path, str]] = []
    for chunk_dir in sorted((src_root / "data").iterdir()):
        if not chunk_dir.is_dir():
            continue
        for parquet in sorted(chunk_dir.glob("episode_*.parquet")):
            out.append((parquet, chunk_dir.name))
    return out


def _probe_image_size(src_parquet: pathlib.Path) -> tuple[int, int, int]:
    """Decode the first frame of the first camera to get the real (H, W, C)."""
    table = pq.read_table(src_parquet)
    first_cam = next(iter(VIDEO_KEY_MAP))
    frame = _decode_image(table[first_cam][0].as_py())
    return frame.shape  # (H, W, C)


def main(
    src: str,
    dst: str,
    *,
    crf: int = 23,
    num_workers: int = 4,
    max_episodes: int = 0,
    overwrite: bool = False,
) -> None:
    src_root = pathlib.Path(src).expanduser().resolve()
    dst_root = pathlib.Path(dst).expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination {dst_root} exists. Pass --overwrite to delete and rewrite."
            )
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    fps = int(src_info["fps"])

    episodes = _iter_chunked_episodes(src_root)
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f"No episode parquets found under {src_root}/data")

    # Derive the true (H, W, C) from a decoded frame: the source feature shape is
    # (C, H, W), and we need (H, W, C) for the video feature + even-dim check.
    image_size = _probe_image_size(episodes[0][0])
    if image_size[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {image_size}")

    print(
        f"[convert] src={src_root} dst={dst_root} episodes={len(episodes)} "
        f"fps={fps} image_size={image_size} num_workers={num_workers} crf={crf}"
    )

    (dst_root / "data").mkdir(parents=True, exist_ok=True)
    (dst_root / "videos").mkdir(parents=True, exist_ok=True)

    kept_indices: set[int] = set()
    total_frames = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_process_episode, parquet, chunk_rel, dst_root, fps, crf): parquet
            for parquet, chunk_rel in episodes
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ep_idx, n = fut.result()
            kept_indices.add(ep_idx)
            total_frames += n
            if i % 20 == 0 or i == len(futures):
                print(f"[convert] {i}/{len(futures)} episodes done ({total_frames} frames)")

    # info.json (rewritten), modality.json (copied), meta jsonl (filtered if subset).
    out_info = _build_target_info(src_info, image_size)
    out_info["total_episodes"] = len(episodes)
    out_info["total_frames"] = total_frames
    out_info["total_videos"] = len(episodes) * len(VIDEO_KEY_MAP)
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta" / "info.json").write_text(json.dumps(out_info, indent=4))
    shutil.copy2(_modality_json_path(), dst_root / "meta" / "modality.json")

    filter_set = kept_indices if max_episodes > 0 else None
    _copy_meta_files(src_root / "meta", dst_root / "meta", kept_episode_indices=filter_set)

    print("[convert] computing stats.json over observation.state + action ...")
    stats = _compute_stats(dst_root)
    (dst_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    print(f"[convert] done. wrote {len(episodes)} episodes, {total_frames} frames -> {dst_root}")


if __name__ == "__main__":
    tyro.cli(main)
