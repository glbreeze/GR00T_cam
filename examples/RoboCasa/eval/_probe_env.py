"""One-off introspection of a RoboCasa v0.2 env (run in the robocasa24 conda env).

Confirms the risk points flagged in openpi's docs/robocasa24_plan.md before we
trust the eval client:
  - env.action_dim and the flat action layout for PandaMobile + OSC_POSE
  - obs dict keys + shapes (camera image keys, the 5 state channels)
  - language access (ep_meta["lang"])
  - rendered image orientation vs the training LeRobot frames
  - which controller loader works under robosuite 1.5.1

Usage:
    python examples/RoboCasa/eval/_probe_env.py --env-name PnPCounterToCab
"""

from __future__ import annotations

import argparse

import numpy as np


def build_env(env_name: str, img_size: int):
    """Mirror robocasa.utils.eval_utils.create_eval_env's robocasa configs but
    use the robosuite-1.5 composite controller loader (the eval_utils one imports
    the removed 1.4 `load_controller_config`)."""
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
        seed=0,
        obj_instance_split="B",
        layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
        translucent_robot=False,
    )
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PnPCounterToCab")
    ap.add_argument("--img-size", type=int, default=128)
    args = ap.parse_args()

    env = build_env(args.env_name, args.img_size)
    print("=== action ===")
    print("action_dim:", env.action_dim)
    low, high = env.action_spec
    print("action_spec low :", np.round(low, 3))
    print("action_spec high:", np.round(high, 3))

    # composite controller action split (names -> slice), if exposed
    try:
        comp = env.robots[0].composite_controller
        print("composite controller:", type(comp).__name__)
        for attr in ("_action_split_indexes", "action_split_indexes"):
            if hasattr(comp, attr):
                print(f"  {attr}:", getattr(comp, attr))
    except Exception as e:  # noqa: BLE001
        print("controller introspection failed:", repr(e))

    obs = env.reset()
    print("\n=== obs keys (shape, dtype) ===")
    for k in sorted(obs):
        v = np.asarray(obs[k])
        tag = "  <-- IMAGE" if k.endswith("_image") else ""
        tag = "  <-- STATE" if any(s in k for s in ("base_pos", "base_quat", "base_to_eef", "gripper_qpos")) else tag
        print(f"  {k:45s} {str(v.shape):18s} {v.dtype}{tag}")

    print("\n=== language ===")
    for getter in ("get_ep_meta",):
        if hasattr(env, getter):
            try:
                ep = getattr(env, getter)()
                print(f"  {getter}()['lang'] = {ep.get('lang')!r}")
            except Exception as e:  # noqa: BLE001
                print(f"  {getter} failed: {e!r}")
    if hasattr(env, "_ep_meta"):
        print("  env._ep_meta['lang'] =", repr(env._ep_meta.get("lang")))

    print("\n=== state assembly (16-dim) ===")
    parts = [
        ("robot0_base_pos", 3),
        ("robot0_base_quat", 4),
        ("robot0_base_to_eef_pos", 3),
        ("robot0_base_to_eef_quat", 4),
        ("robot0_gripper_qpos", 2),
    ]
    total = 0
    for k, d in parts:
        present = k in obs
        sz = int(np.asarray(obs[k]).size) if present else -1
        total += sz if present else 0
        print(f"  {k:30s} present={present} size={sz} (expected {d})")
    print("  TOTAL state dim:", total, "(expected 16)")

    print("\n=== image orientation check ===")
    for k in ("robot0_agentview_left_image", "robot0_eye_in_hand_image"):
        if k in obs:
            im = np.asarray(obs[k])
            print(f"  {k}: shape={im.shape} dtype={im.dtype} "
                  f"min={im.min()} max={im.max()} mean={im.mean():.1f}")

    # success API
    print("\n=== success API ===")
    print("  has _check_success:", hasattr(env, "_check_success"),
          "->", env._check_success() if hasattr(env, "_check_success") else "n/a")

    # step once with a zero action to confirm the 4-tuple contract
    a = np.zeros(env.action_dim, dtype=np.float32)
    out = env.step(a)
    print("\n=== step() return ===")
    print("  len(step output):", len(out), "(robosuite native = 4)")
    env.close()
    print("\nPROBE OK")


if __name__ == "__main__":
    main()
