"""Evaluate a pen-spinning policy in Isaac Sim 6.2 and (optionally) record video + snapshots with RTX.

    python scripts/rollout.py --checkpoint weights/model_8600.pt --video

Writes next to the checkpoint (or --out_dir):
  <name>_eval.json   turns per episode, episode length, drop rate, heading rate
  <name>.mp4         2x2 tiled close-ups of the first four envs (with --video)
  <name>_frame.png   a still from the middle of the video (with --video)
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PenSpin-Sharpa-Play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=None, help="default 4 with --video, else 1024")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--out_dir", default=None)
parser.add_argument("--label", default="")
add_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
if args.video:
    args.enable_cameras = True

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

import pen_spin.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def look_at_quat(eye, target):
    """(x,y,z,w) camera orientation looking from eye to target (world convention: +x forward, +z up)."""
    from scipy.spatial.transform import Rotation

    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    return tuple(Rotation.from_matrix(np.stack([f, left, up], axis=1)).as_quat().tolist())


def light_quat(src, target):
    """(x,y,z,w) orientation for a USD light at ``src`` shining toward ``target`` (lights emit along local -Z)."""
    from scipy.spatial.transform import Rotation

    z = np.asarray(src, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return tuple(Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat().tolist())


def main():
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs or (4 if args.video else 1024)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.seconds + 1.0)
    pen_start = np.array(env_cfg.scene.object.init_state.pos)
    if args.video:
        # per-env close-up camera + a key light for nicer shading
        target = pen_start + np.array([0.025, 0.0, -0.03])
        eye = pen_start + np.array([0.16, -0.24, 0.13])
        env_cfg.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera", height=args.height, width=args.width, data_types=["rgb"],
            update_period=0.0,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=look_at_quat(eye, target), convention="world"),
        )
        # dark studio look: dark slate floor, dim ambient dome, warm key light + cool rim light on the hand
        env_cfg.scene.ground = AssetBaseCfg(
            prim_path="/World/ground", collision_group=-1,
            spawn=sim_utils.GroundPlaneCfg(color=(0.018, 0.02, 0.026)),
        )
        env_cfg.scene.light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=180.0, color=(0.30, 0.33, 0.40))
        )
        env_cfg.scene.key_light = AssetBaseCfg(
            prim_path="/World/KeyLight",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, angle=1.5, color=(1.0, 0.95, 0.88)),
            init_state=AssetBaseCfg.InitialStateCfg(rot=light_quat((0.6, -0.8, 1.2), (0.0, 0.0, 0.0))),
        )
        env_cfg.scene.rim_light = AssetBaseCfg(
            prim_path="/World/RimLight",
            spawn=sim_utils.DistantLightCfg(intensity=1200.0, angle=2.0, color=(0.6, 0.75, 1.0)),
            init_state=AssetBaseCfg.InitialStateCfg(rot=light_quat((-0.8, 0.7, 0.6), (0.0, 0.0, 0.0))),
        )
    with launch_simulation(env_cfg, args):
        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
        )

        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=env_cfg.clip_actions)
        u = env.unwrapped
        ckpt = os.path.abspath(args.checkpoint)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=u.device)

        out_dir = args.out_dir or os.path.dirname(ckpt)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, args.label or os.path.splitext(os.path.basename(ckpt))[0])
        writer = None
        if args.video:
            import cv2
            import imageio.v2 as iio

            cam = u.scene["camera"]
            writer = iio.get_writer(stem + ".mp4", fps=round(1.0 / u.step_dt), quality=8)

        steps = int(args.seconds / u.step_dt)
        ep_turns, ep_lens, drops, heading_rates = [], [], 0, []
        obs = env.get_observations()
        for t in range(steps):
            with torch.inference_mode():
                spin_before = u.spin_angle.clone()
                len_before = u.episode_length_buf.clone()
                obs, _, dones, extras = env.step(policy(obs))
            yr = extras.get("log", {}).get("metric/heading_rate")
            if yr is not None:
                heading_rates.append(float(yr))
            done_ids = dones.nonzero().flatten()
            if len(done_ids):
                ep_turns += (spin_before[done_ids] / (2 * math.pi)).tolist()
                ep_lens += ((len_before[done_ids] + 1).float() * u.step_dt).tolist()
                drops += int(u.reset_terminated[done_ids].sum().item())
            if writer is not None:
                rgb = cam.data.output["rgb"][:4, ..., :3].to(torch.uint8).cpu().numpy()
                tiles = []
                for e in range(rgb.shape[0]):
                    img = np.ascontiguousarray(rgb[e])
                    txt = f"env {e}  t={t * u.step_dt:5.2f}s  turns={u.spin_angle[e].item() / (2 * math.pi):+.2f}"
                    cv2.putText(img, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
                    tiles.append(img)
                while len(tiles) < 4:
                    tiles.append(np.full_like(tiles[0], 15))
                grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:4], 1)], 0)
                writer.append_data(grid)
                if t == steps // 3:
                    iio.imwrite(stem + "_frame.png", grid)
        if writer is not None:
            writer.close()

        ep_turns += (u.spin_angle / (2 * math.pi)).tolist()          # still-running episodes
        ep_lens += (u.episode_length_buf.float() * u.step_dt).tolist()
        res = {
            "checkpoint": ckpt, "num_envs": u.num_envs, "seconds": args.seconds,
            "num_episodes": len(ep_turns),
            "mean_turns_per_episode": float(np.mean(ep_turns)),
            "mean_abs_turns_per_episode": float(np.mean(np.abs(ep_turns))),
            "max_turns_per_episode": float(np.max(ep_turns)),
            "mean_episode_len_s": float(np.mean(ep_lens)),
            "drop_rate_per_episode": drops / len(ep_turns),
            "mean_heading_rate_rad_s": float(np.mean(heading_rates)) if heading_rates else None,
            "gravity": u.gravity,
        }
        print("[rollout] " + json.dumps(res, indent=2), flush=True)
        with open(stem + "_eval.json", "w") as f:
            json.dump(res, f, indent=2)
        if writer is not None:
            print(f"[rollout] wrote {stem}.mp4 and {stem}_frame.png", flush=True)
        env.close()


if __name__ == "__main__":
    main()
