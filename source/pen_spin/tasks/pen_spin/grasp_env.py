"""Grasp-cache generation for pen spinning (Isaac Lab 3.0).

Spawns the pen in the fingers with random yaw + joint noise, holds the hand still (zero actions,
full gravity) and records every state that survives a short episode with >= `min_contacts`
fingertips touching the pen. Rows: [22 joints (ACTUATED_JOINTS order), pen pos (3), pen quat (w,x,y,z)].
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.utils import configclass

from .pen_spin_env import PenSpinEnv, xyzw_to_wxyz
from .pen_spin_env_cfg import PenSpinEnvCfg


@configclass
class PenGraspEnvCfg(PenSpinEnvCfg):
    episode_length_s = 2.0
    gravity_curriculum = False
    grasp_cache_path = ""               # never load a cache while generating one
    reset_joint_noise = 0.25
    reset_yaw_range = math.pi
    random_force_scale = 0.0
    max_tilt = 25.0 * math.pi / 180.0
    max_pen_drift = 0.03
    num_grasps = 50000
    min_contacts = 2
    contact_force_threshold = 0.05      # N
    out_path = "cache/pen_grasps.npy"

    def __post_init__(self):
        super().__post_init__()
        self.events.pen_mass = None     # nominal mass while generating


class PenGraspEnv(PenSpinEnv):
    cfg: PenGraspEnvCfg

    def __init__(self, cfg: PenGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.saved = torch.zeros((0, 29), device=self.device)
        self.done = False

    def save(self, n: int | None = None):
        n = self.saved.shape[0] if n is None else n
        os.makedirs(os.path.dirname(os.path.abspath(self.cfg.out_path)), exist_ok=True)
        np.save(self.cfg.out_path, self.saved[:n].cpu().numpy())
        print(f"[gen_grasp] saved {n} grasps to {self.cfg.out_path}", flush=True)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is not None and len(env_ids) > 0 and not self.done:
            e = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            survived = self.reset_time_outs[e] & ~self.reset_terminated[e]
            forces = torch.norm(self.contact.data.net_forces_w.torch[e], dim=-1)
            good = e[survived & ((forces > self.cfg.contact_force_threshold).sum(-1) >= self.cfg.min_contacts)]
            if len(good) > 0:
                pen_pos = self.object.data.root_pos_w.torch[good] - self.scene.env_origins[good]
                state = torch.cat([self.hand.data.joint_pos.torch[good][:, self.actuated_ids], pen_pos,
                                   xyzw_to_wxyz(self.object.data.root_quat_w.torch[good])], dim=-1)
                self.saved = torch.cat([self.saved, state], dim=0)
                print(f"[gen_grasp] {self.saved.shape[0]} / {self.cfg.num_grasps} (batch yield {len(good)}/{len(e)})", flush=True)
            if self.saved.shape[0] >= self.cfg.num_grasps:
                self.save(self.cfg.num_grasps)
                self.done = True
        super()._reset_idx(env_ids)
