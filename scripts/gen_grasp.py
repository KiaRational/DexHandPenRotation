"""Generate the pen grasp cache (initial states for training), Isaac Lab 3.0.

    python scripts/gen_grasp.py --num_envs 8192 --out cache/pen_grasps.npy
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PenSpin-Sharpa-Grasp")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--num_grasps", type=int, default=50000)
parser.add_argument("--out", default="cache/pen_grasps.npy")
parser.add_argument("--max_steps", type=int, default=3000)
add_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pen_spin.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.num_grasps = args.num_grasps
cfg.out_path = args.out
with launch_simulation(cfg, args):
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    env.reset()
    zeros = torch.zeros((u.num_envs, cfg.action_space), device=u.device)
    for _ in range(args.max_steps):
        env.step(zeros)
        if u.done:
            break
    else:
        print(f"[gen_grasp] stopped after {args.max_steps} steps with {u.saved.shape[0]} grasps", flush=True)
        if u.saved.shape[0] >= 200:
            u.save()
    env.close()
