"""LIBERO benchmark eval client for GR00T N1.5 (CamVLA-aware).

Talks to a gr00t inference server over ZMQ (see ``scripts/inference_service.py``)
and drives LIBERO rollouts via robosuite/mujoco. Mirrors the structure of
``openpi/examples/libero/main.py`` (task-id sharding, action-chunk replanning,
JSON summary) but serializes gr00t-format obs dicts (``state.x/y/z/...``,
``video.image``, ``video.wrist_image``, ``annotation.human...``).

When ``--use-camera-params`` is on, also sends paired camera intrinsics and
extrinsics under the keys :class:`CameraAwareLeRobotDataset` emits::

    camera.agent_intrinsic   (1, 3, 3)
    camera.agent_extrinsic   (1, 4, 4)
    camera.wrist_intrinsic   (1, 3, 3)
    camera.wrist_extrinsic   (1, 4, 4)

Intrinsics are computed at the LIBERO render resolution (256x256), which is
also what we send to the server. The server's ``VideoResizeCameraAware``
rescales K when it resizes the image to the model's 224x224 grid.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import pathlib
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro


# --- LIBERO config bootstrap ------------------------------------------------
# The libero submodule's __init__.py drops into an interactive input() prompt
# on first import if no config.yaml exists. Pre-populate one (honoring
# LIBERO_CONFIG_PATH so the sbatch can point at per-job scratch) before we
# import any libero symbol.
_LIBERO_CONFIG_DIR = pathlib.Path(
    os.environ.get("LIBERO_CONFIG_PATH", str(pathlib.Path.home() / ".libero"))
)
_LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["LIBERO_CONFIG_PATH"] = str(_LIBERO_CONFIG_DIR)
_LIBERO_CONFIG_FILE = _LIBERO_CONFIG_DIR / "config.yaml"
if not _LIBERO_CONFIG_FILE.exists():
    _LIBERO_SRC = pathlib.Path(
        os.environ.get(
            "LIBERO_SRC_ROOT",
            "/scratch/lg154/Research/openpi/third_party/libero/libero/libero",
        )
    )
    _LIBERO_CONFIG_FILE.write_text(
        "\n".join(
            [
                f"benchmark_root: {_LIBERO_SRC}",
                f"bddl_files: {_LIBERO_SRC / 'bddl_files'}",
                f"init_states: {_LIBERO_SRC / 'init_files'}",
                f"datasets: {_LIBERO_SRC.parent / 'datasets'}",
                f"assets: {_LIBERO_SRC / 'assets'}",
            ]
        )
        + "\n"
    )

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from examples.Libero.eval.utils import (  # noqa: E402
    get_libero_image,
    quat2axisangle,
)


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")

# Per-suite step budget (longest training demo + headroom). Matches openpi.
_MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    """Command-line config for ``run_libero_eval.py``."""

    # ---- inference server (talks ZMQ to scripts/inference_service.py) ----
    host: str = "localhost"
    port: int = 19200
    model_name: str = ""  # labels output dirs / JSON summaries

    # ---- LIBERO task / suite ----
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    task_id_start: int = 0
    task_id_end: int = -1  # exclusive; -1 = all tasks in suite

    # ---- action chunking ----
    # The server returns 16 actions per inference (action_indices=range(16)).
    # We execute the first `replan_steps` of them, then re-query.
    replan_steps: int = 5

    # ---- CamVLA path ----
    # When True, also send camera.{agent,wrist}_{intrinsic,extrinsic} in obs.
    # Server must be running a CamVLA-capable data config / checkpoint.
    use_camera_params: bool = False

    # ---- output ----
    video_out_path: Optional[str] = None  # default: ./rollouts/<suite>
    summary_out_path: Optional[str] = None  # JSON summary; default: none

    # ---- misc ----
    seed: int = 7


# --- camera helpers (ported from openpi/examples/libero/main.py) ------------
def _get_camera_extrinsic(env, camera_name: str) -> np.ndarray:
    """4x4 [R|t] in world frame from MuJoCo cam_xpos/cam_xmat."""
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(env.sim.data.cam_xpos[cam_id], dtype=np.float32)
    cam_rot = np.asarray(env.sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = cam_rot
    extrinsic[:3, 3] = cam_pos
    return extrinsic


def _get_camera_intrinsic(env, camera_name: str, image_h: int, image_w: int) -> np.ndarray:
    """Pinhole K from MuJoCo ``cam_fovy`` for an ``image_h x image_w`` render.

    Must be evaluated at the resolution actually sent to the policy: the
    server's ``VideoResizeCameraAware`` only rescales K when it also resizes
    the image, so a 256-px obs needs K at 256-px.
    """
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy_rad = np.deg2rad(float(env.sim.model.cam_fovy[cam_id]))
    fy = (image_h / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy
    cx = image_w / 2.0
    cy = image_h / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


# --- env / obs / action ------------------------------------------------------
def _make_env(task, resolution: int, seed: int):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _load_task_init_states(task):
    """Load init-states with ``weights_only=False``.

    LIBERO's ``benchmark.get_task_init_states`` calls ``torch.load`` without that
    flag; on torch >= 2.6 (openpi venv) it rejects the pickled numpy arrays in
    the init-state files. Mirrors openpi/examples/libero/main.py.
    """
    import torch

    init_states_path = (
        pathlib.Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(init_states_path, weights_only=False)


def _build_obs(obs, lang: str, env, use_camera_params: bool) -> dict:
    """Convert a LIBERO obs into the gr00t modality dict."""
    img, wrist = get_libero_image(obs)  # rotated 180° to match training data
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    grip = obs["robot0_gripper_qpos"]

    out = {
        "video.image": np.expand_dims(img, 0),
        "video.wrist_image": np.expand_dims(wrist, 0),
        "state.x": np.array([[xyz[0]]], dtype=np.float32),
        "state.y": np.array([[xyz[1]]], dtype=np.float32),
        "state.z": np.array([[xyz[2]]], dtype=np.float32),
        "state.roll": np.array([[rpy[0]]], dtype=np.float32),
        "state.pitch": np.array([[rpy[1]]], dtype=np.float32),
        "state.yaw": np.array([[rpy[2]]], dtype=np.float32),
        "state.gripper": np.expand_dims(np.asarray(grip, dtype=np.float32), 0),
        "annotation.human.action.task_description": [lang],
    }

    if use_camera_params:
        H = W = LIBERO_ENV_RESOLUTION
        out["camera.agent_intrinsic"] = _get_camera_intrinsic(env, "agentview", H, W)[None]
        out["camera.wrist_intrinsic"] = _get_camera_intrinsic(env, "robot0_eye_in_hand", H, W)[None]
        out["camera.agent_extrinsic"] = _get_camera_extrinsic(env, "agentview")[None]
        out["camera.wrist_extrinsic"] = _get_camera_extrinsic(env, "robot0_eye_in_hand")[None]

    return out


def _extract_action_chunk(action_dict: dict, replan_steps: int) -> np.ndarray:
    """Stack the per-key 16-step server response into a ``(replan_steps, 7)`` chunk.

    The policy returns each ``action.<key>`` with shape ``(action_horizon, dim)``
    after the batch-dim squeeze in `policy.get_action`. Every action component is
    1-D in the LIBERO modality.json, so we ``reshape(-1)`` to drop the trailing
    singleton before stacking — `np.stack([..., axis=-1])` of `(N, 1)` arrays
    would otherwise produce a 3-D `(N, 1, K)` tensor.

    Gripper is NOT post-processed: training data action.gripper is already in
    robosuite's {-1, +1} convention, and the server's min_max denorm preserves
    that range. openpi's main.py also passes its action chunk through unchanged.
    """
    return np.stack(
        [
            np.asarray(action_dict[f"action.{k}"], dtype=np.float32).reshape(-1)[:replan_steps]
            for k in ACTION_KEYS
        ],
        axis=-1,
    )


# --- main eval loop ----------------------------------------------------------
def eval_libero(args: Args):
    np.random.seed(args.seed)

    from gr00t.eval.service import ExternalRobotInferenceClient  # noqa: E402

    client = ExternalRobotInferenceClient(host=args.host, port=args.port)

    bd = benchmark.get_benchmark_dict()
    suite = bd[args.task_suite_name]()
    n_tasks = suite.n_tasks
    task_start = max(0, args.task_id_start)
    task_end = n_tasks if args.task_id_end < 0 else min(args.task_id_end, n_tasks)
    if task_start >= task_end:
        raise ValueError(
            f"Invalid task range [{task_start}, {task_end}) for suite with {n_tasks} tasks"
        )
    if args.task_suite_name not in _MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = _MAX_STEPS_BY_SUITE[args.task_suite_name]

    video_dir = pathlib.Path(args.video_out_path or f"./rollouts/{args.task_suite_name}")
    video_dir.mkdir(parents=True, exist_ok=True)
    log_path = video_dir / "eval.log"
    log_file = log_path.open("w")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(
        f"Task suite: {args.task_suite_name}   "
        f"tasks: [{task_start}, {task_end})   trials/task: {args.num_trials_per_task}"
    )
    log(
        f"use_camera_params={args.use_camera_params}   "
        f"replan_steps={args.replan_steps}   server={args.host}:{args.port}"
    )

    total_episodes, total_successes = 0, 0
    records: list[dict] = []

    for task_id in tqdm.tqdm(range(task_start, task_end), desc="tasks"):
        task = suite.get_task(task_id)
        init_states = _load_task_init_states(task)
        env, task_description = _make_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for ep in tqdm.tqdm(
            range(args.num_trials_per_task), desc=f"task {task_id}", leave=False
        ):
            log(f"[task {task_id}] {task_description}  episode {ep + 1}")
            env.reset()
            obs = env.set_init_state(init_states[ep])

            action_plan: collections.deque = collections.deque()
            top_view, wrist_view = [], []
            done = False
            t = 0
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _r, _d, _i = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img, wrist_img = get_libero_image(obs)
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    if not action_plan:
                        obs_dict = _build_obs(
                            obs, task_description, env, args.use_camera_params
                        )
                        action_dict = client.get_action(obs_dict)
                        chunk = _extract_action_chunk(action_dict, args.replan_steps)
                        if chunk.shape[0] < args.replan_steps:
                            raise RuntimeError(
                                f"Server returned only {chunk.shape[0]} actions; "
                                f"need {args.replan_steps}"
                            )
                        action_plan.extend(chunk)

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    log(f"  [exception] {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "fail"
            mp4_path = video_dir / f"task{task_id:03d}_ep{ep:02d}_{suffix}.mp4"
            with imageio.get_writer(mp4_path, fps=30) as writer:
                for a, b in zip(top_view, wrist_view):
                    writer.append_data(np.hstack((a, b)))

            log(
                f"  -> success={done}  total {total_successes}/{total_episodes}  "
                f"({100.0 * total_successes / max(1, total_episodes):.1f}%)"
            )

        records.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_successes / max(1, task_episodes),
            }
        )
        log(f"  task {task_id} rate: {task_successes}/{task_episodes}")

    summary = {
        "model_name": args.model_name or None,
        "task_suite_name": args.task_suite_name,
        "task_range": [task_start, task_end],
        "num_trials_per_task": args.num_trials_per_task,
        "use_camera_params": args.use_camera_params,
        "replan_steps": args.replan_steps,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_successes / max(1, total_episodes),
        "records": records,
    }
    if args.summary_out_path:
        out = pathlib.Path(args.summary_out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        log(f"Summary written to {out}")

    log(
        f"DONE  total {total_successes}/{total_episodes}  "
        f"({100.0 * total_successes / max(1, total_episodes):.1f}%)"
    )
    log_file.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Parse the dataclass directly so flags land at the top level (--summary-out-path),
    # not behind an --args. prefix. tyro infers the prefix from the function-parameter
    # name otherwise.
    eval_libero(tyro.cli(Args))
