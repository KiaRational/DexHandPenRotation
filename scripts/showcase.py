"""README visuals for a trained pen-spinning policy (Isaac Sim 6.2 RTX, dark studio look).

    python scripts/showcase.py --checkpoint logs/rsl_rl/.../model_8600.pt --out_dir media/showcase

Writes to --out_dir:
  hero.mp4 / hero.gif            close-up of one hand spinning the pen
  overview.mp4 / overview.gif    3x3 mosaic of hands spinning in parallel
  hero_still.png, overview_still.png
  motion_trail.png               "long exposure" of one revolution
  turns_over_time.png            cumulative pen turns of every hand vs time
"""

import argparse
import math
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PenSpin-Sharpa-Play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out_dir", default="media/showcase")
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--hero_size", type=int, default=900)
parser.add_argument("--gif_size", type=int, default=480)
parser.add_argument("--gif_seconds", type=float, default=6.0)
add_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.enable_cameras = True

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

import pen_spin.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def _rot(cols):
    from scipy.spatial.transform import Rotation

    return tuple(Rotation.from_matrix(np.stack(cols, axis=1)).as_quat().tolist())


def look_at_quat(eye, target):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    return _rot([f, left, np.cross(f, left)])


def light_quat(src, target):
    z = np.asarray(src, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    return _rot([x, np.cross(z, x), z])


def save_gif(frames, path, size, fps):
    from PIL import Image

    imgs = []
    for f in frames:
        im = Image.fromarray(f).resize((size, round(size * f.shape[0] / f.shape[1])), Image.LANCZOS)
        imgs.append(im.quantize(colors=192, method=Image.MEDIANCUT, dither=Image.NONE))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f"[showcase] wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def main():
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.seconds + 5.0
    pen = np.array(env_cfg.scene.object.init_state.pos)
    sp = env_cfg.scene.env_spacing

    # hero close-up: low three-quarter view, slightly from the fingertip side
    eye = pen + np.array([0.13, -0.19, 0.07])
    tgt = pen + np.array([0.035, 0.0, -0.035])
    env_cfg.scene.camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/HeroCam", height=args.hero_size, width=args.hero_size, data_types=["rgb"],
        update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=look_at_quat(eye, tgt), convention="world"),
    )
    # dark studio
    env_cfg.scene.ground = AssetBaseCfg(
        prim_path="/World/ground", collision_group=-1, spawn=sim_utils.GroundPlaneCfg(color=(0.012, 0.014, 0.018)),
    )
    env_cfg.scene.light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=140.0, color=(0.28, 0.31, 0.40)),
    )
    env_cfg.scene.key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=3200.0, angle=1.2, color=(1.0, 0.94, 0.86)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=light_quat((0.6, -0.8, 1.2), (0.0, 0.0, 0.0))),
    )
    env_cfg.scene.rim_light = AssetBaseCfg(
        prim_path="/World/RimLight",
        spawn=sim_utils.DistantLightCfg(intensity=1600.0, angle=2.0, color=(0.55, 0.72, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=light_quat((-0.8, 0.7, 0.6), (0.0, 0.0, 0.0))),
    )

    with launch_simulation(env_cfg, args):
        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
        )
        import imageio.v2 as iio

        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=env_cfg.clip_actions)
        u = env.unwrapped
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(os.path.abspath(args.checkpoint))
        policy = runner.get_inference_policy(device=u.device)

        out = args.out_dir
        os.makedirs(out, exist_ok=True)
        fps = round(1.0 / u.step_dt)
        hero_w = iio.get_writer(f"{out}/hero.mp4", fps=fps, quality=9)
        ov_w = iio.get_writer(f"{out}/overview.mp4", fps=fps, quality=9)
        hero_cam = u.scene["camera"]
        import cv2

        steps = int(args.seconds / u.step_dt)
        warm = int(1.0 / u.step_dt)                       # skip the first second (grasp settling) in GIFs
        n_gif = int(args.gif_seconds / u.step_dt)
        hero_frames, ov_frames, turns, done_any = [], [], [], torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
        obs = env.get_observations()
        for t in range(steps):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            done_any |= dones.bool()
            turns.append((u.spin_angle / (2 * math.pi)).cpu().numpy().copy())
            h = hero_cam.data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()
            allr = hero_cam.data.output["rgb"][:, ..., :3].to(torch.uint8).cpu().numpy()
            n = int(math.ceil(math.sqrt(len(allr))))
            ts_ = args.hero_size // n
            tiles = [cv2.resize(x, (ts_, ts_), interpolation=cv2.INTER_AREA) for x in allr]
            tiles += [np.full_like(tiles[0], 12)] * (n * n - len(tiles))
            o = np.concatenate([np.concatenate(tiles[r * n:(r + 1) * n], 1) for r in range(n)], 0)
            o = np.pad(o, ((0, args.hero_size - o.shape[0]), (0, args.hero_size - o.shape[1]), (0, 0)))
            hero_w.append_data(h)
            ov_w.append_data(o)
            if warm <= t < warm + n_gif:
                hero_frames.append(h)
                ov_frames.append(o)
        hero_w.close()
        ov_w.close()
        print(f"[showcase] wrote hero.mp4, overview.mp4; drops={int(done_any.sum())}/{u.num_envs}", flush=True)

        turns = np.array(turns)
        iio.imwrite(f"{out}/hero_still.png", hero_frames[len(hero_frames) // 2])
        iio.imwrite(f"{out}/overview_still.png", ov_frames[len(ov_frames) // 2])
        save_gif(hero_frames[::2], f"{out}/hero.gif", args.gif_size, fps / 2)
        save_gif(ov_frames[::2], f"{out}/overview.gif", args.gif_size, fps / 2)

        # motion trail: lighten-blend frames spanning one revolution of env 0's pen
        t0 = warm + n_gif // 3
        tt = turns[:, 0]
        t1 = t0 + int(np.argmax(tt[t0:] - tt[t0] >= 0.5) or fps)
        idx = np.linspace(t0 - warm, min(t1 - warm, len(hero_frames) - 1), 9).astype(int)
        stack = np.stack([hero_frames[i].astype(np.float32) for i in idx])
        trail = 0.55 * stack.max(0) + 0.45 * stack[-1]
        iio.imwrite(f"{out}/motion_trail.png", trail.clip(0, 255).astype(np.uint8))

        # cumulative turns plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = np.arange(len(turns)) * u.step_dt
        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        cmap = plt.get_cmap("cool")
        for e in range(turns.shape[1]):
            ax.plot(ts, turns[:, e], color=cmap(e / max(1, turns.shape[1] - 1)), lw=1.6, alpha=0.9)
        ax.plot(ts, turns.mean(1), color="white", lw=2.6, label=f"mean ({turns[-1].mean():.1f} turns in {ts[-1]:.0f} s)")
        ax.set_xlabel("time [s]", color="#c9d1d9")
        ax.set_ylabel("pen turns", color="#c9d1d9")
        ax.set_title(f"Pen revolutions of {turns.shape[1]} hands, full gravity, deterministic policy",
                     color="#e6edf3", fontsize=11)
        ax.tick_params(colors="#8b949e")
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.grid(color="#21262d", lw=0.8)
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", loc="upper left")
        fig.tight_layout()
        fig.savefig(f"{out}/turns_over_time.png", facecolor=fig.get_facecolor())
        print(f"[showcase] done: {out}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
