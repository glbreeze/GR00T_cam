"""RoboCasa v0.2 (24-task) eval client for GR00T N1.5 (CamVLA-aware).

Counterpart to ``examples/Libero/eval/run_libero_eval.py``: talks to a gr00t
inference server over ZMQ (``scripts/inference_service.py --server``) and drives
RoboCasa v0.2 rollouts via robosuite/mujoco. Must run in the ``robocasa24`` conda
env (robosuite 1.5.1 + robocasa v0.2 + kitchen assets); the server runs in the
``gr00t`` env. Both talk over ZMQ on localhost.

RoboCasa v0.2 differences vs LIBERO (which is why this is a separate client):
  - robosuite **native** API: ``env.reset()`` returns an obs dict and
    ``env.step()`` returns a 4-tuple ``(obs, reward, done, info)`` (no gym
    wrapper, unlike openpi's robocasa365 path).
  - PandaMobile robot, 11-dim native action (server returns 12-dim; we drop
    control_mode), 16-dim state, two cameras.
  - No registered env init-states: each rollout is a fresh ``env.reset(seed)``
    that samples a new kitchen layout/object (``obj_instance_split="B"`` =
    held-out eval objects), and the language comes from ``env.get_ep_meta()``.

Action mapping. The server returns a 12-dim action in GR00T/LeRobot order
``[base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]`` (see
examples/RoboCasa/modality.json). The PandaMobile composite controller, however,
exposes an **11-dim** native action (verified by probing the live env):
``_action_split_indexes = right(0:6), right_gripper(6:7), base(7:10),
torso(10:11)`` — i.e. ``[eef_pos(3), eef_rot(3), gripper(1), base(3),
torso(1)]`` with **no** control_mode/base_mode dim. So we drop LeRobot idx 4
(control_mode) entirely and gather ``native = lerobot[[5,6,7,8,9,10,11,0,1,2,3]]``
(base_motion(4) -> base(3)+torso(1)). The v0 eval fixes from openpi's
robocasa24_plan.md (gripper threshold, base-motion deadzone) are applied in
LeRobot order before the gather; the control-mode lock is a no-op here since
that dim is dropped.
"""

from __future__ import annotations

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

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Robosuite camera names → gr00t video keys. agent=front, wrist=eye-in-hand
# (matches the converter's column rename; see examples/RoboCasa/modality.json).
CAM_KEYS = {
    "robot0_agentview_left_image": "video.image",
    "robot0_eye_in_hand_image": "video.wrist_image",
}
# Robosuite obs key → gr00t state key (+ expected dim), concatenated to 16-dim.
STATE_KEYS = (
    ("robot0_base_pos", "state.base_pos", 3),
    ("robot0_base_quat", "state.base_quat", 4),
    ("robot0_base_to_eef_pos", "state.base_to_eef_pos", 3),
    ("robot0_base_to_eef_quat", "state.base_to_eef_quat", 4),
    ("robot0_gripper_qpos", "state.gripper_qpos", 2),
)
# Server action.* keys in GR00T/LeRobot order; concatenated to a 12-dim vector.
ACTION_KEYS = (
    ("action.base_motion", 4),
    ("action.control_mode", 1),
    ("action.eef_pos", 3),
    ("action.eef_rot", 3),
    ("action.gripper", 1),
)
# LeRobot(12) -> native env(11) gather indices. control_mode (idx 4) is DROPPED:
# native[0:6]=eef_pos+eef_rot=lerobot[5:11], native[6]=gripper=lerobot[11],
# native[7:10]=base=lerobot[0:3], native[10]=torso=lerobot[3].
LEROBOT_TO_NATIVE = np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3], dtype=np.int64)

# The 24 RoboCasa v0.2 atomic task families (the converted dataset's coverage;
# drops NavigateKitchen + the 5 multi-stage composites from the registry).
ROBOCASA_24_TASKS = (
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter",
    "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "PnPCounterToStove",
    "PnPStoveToCounter", "OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor",
    "CloseDoubleDoor", "OpenDrawer", "CloseDrawer", "TurnOnSinkFaucet",
    "TurnOffSinkFaucet", "TurnSinkSpout", "TurnOnStove", "TurnOffStove",
    "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton", "TurnOnMicrowave",
    "TurnOffMicrowave",
)


@dataclasses.dataclass
class Args:
    """Command-line config for ``run_robocasa_eval.py``."""

    # ---- inference server (ZMQ to scripts/inference_service.py) ----
    host: str = "localhost"
    port: int = 19300
    model_name: str = ""  # labels output dirs / JSON summaries

    # ---- task ----
    env_name: str = "PnPCounterToCab"  # one of ROBOCASA_24_TASKS
    num_trials: int = 50
    max_steps: int = 500

    # ---- action chunking ----
    # Server returns 16 actions (action_indices=range(16)). Execute the first
    # `replan_steps`, then re-query.
    replan_steps: int = 16

    # ---- CamVLA path ----
    use_camera_params: bool = False
    img_size: int = 128  # render res; server resizes to 224 (matches training)

    # ---- v0 eval action fixes (mirror openpi/robocasa24_plan.md) ----
    lock_control_mode: bool = True       # no-op: control_mode dim is dropped (11-dim env)
    gripper_threshold: float = 0.0       # binarize gripper at this value
    binarize_gripper: bool = True
    base_motion_deadzone: float = 0.02   # zero base/torso dims with |v| < this

    # ---- output ----
    video_out_path: Optional[str] = None
    summary_out_path: Optional[str] = None
    video_every: int = 10  # write a rollout mp4 every N episodes (0 = none)

    # ---- misc ----
    seed: int = 0


# --- env construction (mirrors create_eval_env, robosuite-1.5 controller) ----
def _make_env(env_name: str, img_size: int, seed: int):
    """Build a RoboCasa v0.2 eval env. Mirrors
    ``robocasa.utils.eval_utils.create_eval_env`` (PandaMobile, OSC_POSE,
    obj_instance_split="B", fixed eval layouts) but uses the robosuite-1.5
    composite controller loader (eval_utils imports the removed 1.4 loader)."""
    import robosuite
    import robosuite_models  # noqa: F401  registers PandaMobile
    import robocasa  # noqa: F401  registers kitchen envs
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(controller=None, robot="PandaMobile")
    env = robosuite.make(
        env_name=env_name,
        robots="PandaMobile",
        controller_configs=controller_config,
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        camera_widths=img_size,
        camera_heights=img_size,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split="B",
        layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
        translucent_robot=False,
    )
    return env


def _episode_language(env) -> str:
    """Task instruction for the current episode (set on reset)."""
    try:
        lang = env.get_ep_meta().get("lang")
    except Exception:  # noqa: BLE001
        lang = getattr(env, "_ep_meta", {}).get("lang")
    return str(lang or "").strip()


# --- camera helpers (MuJoCo cam_xpos/cam_xmat/cam_fovy) ----------------------
def _camera_extrinsic(env, cam_name: str) -> np.ndarray:
    cid = env.sim.model.camera_name2id(cam_name)
    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = np.asarray(env.sim.data.cam_xmat[cid], dtype=np.float32).reshape(3, 3)
    ext[:3, 3] = np.asarray(env.sim.data.cam_xpos[cid], dtype=np.float32)
    return ext


def _camera_intrinsic(env, cam_name: str, h: int, w: int) -> np.ndarray:
    cid = env.sim.model.camera_name2id(cam_name)
    fovy = np.deg2rad(float(env.sim.model.cam_fovy[cid]))
    fy = (h / 2.0) / np.tan(fovy / 2.0)
    return np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


# --- obs / action ------------------------------------------------------------
def _build_obs(obs: dict, lang: str, env, use_camera_params: bool) -> dict:
    """RoboCasa obs dict -> gr00t modality dict (leading batch dim of 1)."""
    out: dict = {"annotation.human.action.task_description": [lang]}
    for cam_key, vid_key in CAM_KEYS.items():
        # robosuite obs images share the orientation of the training LeRobot
        # frames (the converter read them straight from the source HDF5), so
        # send them unflipped.
        out[vid_key] = np.asarray(obs[cam_key], dtype=np.uint8)[None]
    for obs_key, state_key, _dim in STATE_KEYS:
        out[state_key] = np.asarray(obs[obs_key], dtype=np.float32).reshape(1, -1)
    if use_camera_params:
        h = w = env.camera_heights[0] if isinstance(env.camera_heights, (list, tuple)) else env.camera_heights
        out["camera.agent_intrinsic"] = _camera_intrinsic(env, "robot0_agentview_left", h, w)[None]
        out["camera.wrist_intrinsic"] = _camera_intrinsic(env, "robot0_eye_in_hand", h, w)[None]
        out["camera.agent_extrinsic"] = _camera_extrinsic(env, "robot0_agentview_left")[None]
        out["camera.wrist_extrinsic"] = _camera_extrinsic(env, "robot0_eye_in_hand")[None]
    return out


def _extract_action_chunk(action_dict: dict, replan_steps: int) -> np.ndarray:
    """Stack the server's per-key response into a ``(replan_steps, 12)`` chunk in
    LeRobot order. Each ``action.<key>`` comes back as ``(action_horizon, dim)``
    (or ``(action_horizon,)`` for the 1-D channels after the batch squeeze)."""
    cols = []
    for key, _dim in ACTION_KEYS:
        a = np.asarray(action_dict[key], dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        cols.append(a)
    chunk = np.concatenate(cols, axis=-1)  # (action_horizon, 12)
    assert chunk.shape[-1] == 12, f"expected 12-dim action, got {chunk.shape}"
    return chunk[:replan_steps]


def _lerobot_to_native(action: np.ndarray, args: Args) -> np.ndarray:
    """Apply v0 eval fixes (in LeRobot order) then gather to the 11-dim native
    env order. control_mode (idx 4) is dropped by the gather, so
    ``lock_control_mode`` has no effect on the stepped action."""
    a = np.asarray(action, dtype=np.float32).copy()
    if args.base_motion_deadzone > 0:
        bm = a[0:4]  # base_motion -> base(3)+torso(1)
        bm[np.abs(bm) < args.base_motion_deadzone] = 0.0
        a[0:4] = bm
    if args.binarize_gripper:
        a[11] = -1.0 if a[11] < args.gripper_threshold else 1.0
    return a[LEROBOT_TO_NATIVE]


# --- main eval loop ----------------------------------------------------------
def eval_robocasa(args: Args) -> dict:
    np.random.seed(args.seed)
    if args.env_name not in ROBOCASA_24_TASKS:
        logging.warning("env_name %r is not in the known 24-task list", args.env_name)

    from gr00t.eval.service import ExternalRobotInferenceClient  # noqa: E402

    client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    env = _make_env(args.env_name, args.img_size, args.seed)
    assert env.action_dim == 11, f"expected 11-dim action env, got {env.action_dim}"

    video_dir = pathlib.Path(args.video_out_path or f"./rollouts/{args.env_name}")
    video_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    episodes = []
    for ep in tqdm.tqdm(range(args.num_trials), desc=args.env_name):
        obs = env.reset()
        lang = _episode_language(env)
        save_video = args.video_every > 0 and (ep % args.video_every == 0)
        frames = []
        plan = np.empty((0, 12), dtype=np.float32)
        plan_i = 0
        success = False
        step = 0
        while step < args.max_steps:
            if plan_i >= len(plan):
                action_dict = client.get_action(_build_obs(obs, lang, env, args.use_camera_params))
                plan = _extract_action_chunk(action_dict, args.replan_steps)
                plan_i = 0
            native = _lerobot_to_native(plan[plan_i], args)
            plan_i += 1
            if save_video:
                frames.append(np.asarray(obs["robot0_agentview_left_image"], dtype=np.uint8))
            obs, _reward, _done, _info = env.step(native)
            step += 1
            if env._check_success():
                success = True
                break

        successes += int(success)
        episodes.append({"episode": ep, "success": bool(success), "steps": step, "prompt": lang})
        if save_video and frames:
            mp4 = video_dir / f"ep{ep:03d}_{'success' if success else 'fail'}.mp4"
            try:
                imageio.mimsave(mp4, frames, fps=20)
            except Exception:  # noqa: BLE001
                logging.exception("video write failed for %s", mp4)
        running = successes / (ep + 1)
        print(json.dumps({"env": args.env_name, "episode": ep, "success": bool(success),
                          "steps": step, "running_success_rate": running}), flush=True)

    rate = successes / max(1, args.num_trials)
    summary = {
        "model_name": args.model_name or None,
        "env_name": args.env_name,
        "num_trials": args.num_trials,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "use_camera_params": args.use_camera_params,
        "total_episodes": len(episodes),
        "total_successes": successes,
        "success_rate": rate,
        "episodes": episodes,
    }
    if args.summary_out_path:
        out = pathlib.Path(args.summary_out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Summary written to {out}", flush=True)
    print(f"DONE {args.env_name}: {successes}/{len(episodes)} ({100 * rate:.1f}%)", flush=True)
    env.close()
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_robocasa(tyro.cli(Args))
