"""Pen spinning environment (DirectRLEnv, Isaac Lab 3.0) for the Sharpa Wave hand.

Task: keep a pen roughly horizontal in the fingers and spin it about the world z-axis
(heading rate of the pen axis) as fast as possible (clipped) without dropping it.

Internal quaternions are Isaac Lab 3.0's (x, y, z, w). The grasp cache and the policy
observation use (w, x, y, z).
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, saturate


from .pen_spin_env_cfg import PenSpinEnvCfg


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q[..., [1, 2, 3, 0]]


class PenSpinEnv(DirectRLEnv):
    cfg: PenSpinEnvCfg

    def __init__(self, cfg: PenSpinEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.hand, self.object, self.contact = (self.scene[k] for k in ("robot", "object", "contact"))
        n, dev = self.num_envs, self.device

        ids, _ = self.hand.find_joints(cfg.actuated_joint_names, preserve_order=True)
        assert len(ids) == len(cfg.actuated_joint_names), "actuated joints not found"
        self.actuated_ids = torch.tensor(ids, dtype=torch.long, device=dev)
        tips, _ = self.hand.find_bodies(cfg.fingertip_body_names, preserve_order=True)
        self.fingertip_ids = torch.tensor(tips, dtype=torch.long, device=dev)
        self.num_dofs = self.hand.num_joints

        limits = self.hand.data.joint_limits.torch.to(dev)
        self.dof_lower = limits[..., 0] * cfg.dof_limits_scale
        self.dof_upper = limits[..., 1] * cfg.dof_limits_scale
        self.act_lower = self.dof_lower[:, self.actuated_ids]
        self.act_upper = self.dof_upper[:, self.actuated_ids]
        self.default_dof_pos = self.hand.data.default_joint_pos.torch.clone()

        self.cur_targets = self.default_dof_pos.clone()
        self.actions = torch.zeros((n, cfg.action_space), device=dev)
        self.prev_actions = torch.zeros_like(self.actions)

        self.pen_start_pos = torch.zeros((n, 3), device=dev)
        self.pen_prev_yaw = torch.zeros(n, device=dev)
        self.spin_angle = torch.zeros(n, device=dev)      # accumulated signed heading rotation (rad)
        self.rb_forces = torch.zeros((n, 1, 3), device=dev)
        self.pen_local_axis = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(n, 1)

        self.grasp_cache = None
        if cfg.grasp_cache_path and os.path.isfile(cfg.grasp_cache_path):
            g = torch.from_numpy(np.load(cfg.grasp_cache_path)).float().to(dev)
            g[:, 25:29] = wxyz_to_xyzw(g[:, 25:29])
            self.grasp_cache = g
            print(f"[PenSpin] loaded {g.shape[0]} grasps from {cfg.grasp_cache_path}")
        else:
            print("[PenSpin] no grasp cache found, resetting from default pose + noise")

        self.gravity = cfg.gravity_start if cfg.gravity_curriculum else abs(cfg.sim.gravity[2])
        self._set_gravity(self.gravity)
        self.drop_rate_ema = 1.0

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-self.cfg.clip_actions, self.cfg.clip_actions).clone()
        ids = self.actuated_ids
        targets = self.cur_targets[:, ids] + self.cfg.action_scale * self.actions
        self.cur_targets[:, ids] = saturate(targets, self.act_lower, self.act_upper)

        # random perturbation forces on the pen (robustness)
        if self.cfg.random_force_scale > 0:
            self.rb_forces *= 0.9
            hit = torch.rand(self.num_envs, device=self.device) < self.cfg.random_force_prob
            mass = self.object.data.body_mass.torch.view(self.num_envs, 1).to(self.device)
            new = torch.randn((self.num_envs, 3), device=self.device) * mass * self.gravity * self.cfg.random_force_scale
            self.rb_forces[hit, 0] = new[hit]
            self.object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self.rb_forces, torques=torch.zeros_like(self.rb_forces), is_global=True
            )

    def _apply_action(self) -> None:
        self.hand.actuators.target_command.set_position_index(
            value=self.cur_targets[:, self.actuated_ids], joint_ids=self.actuated_ids
        )

    # ----------------------------------------------------------- observations
    def _pen_state(self):
        pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        quat = self.object.data.root_quat_w.torch            # (x,y,z,w)
        axis = quat_apply(quat, self.pen_local_axis)
        return pos, quat, axis

    def _get_observations(self) -> dict:
        ids = self.actuated_ids
        q = self.hand.data.joint_pos.torch[:, ids]
        lo, hi = self.act_lower, self.act_upper
        q_n = 2.0 * (q - lo) / (hi - lo) - 1.0
        q_n = q_n + self.cfg.joint_obs_noise * torch.randn_like(q_n)
        qd = 0.1 * self.hand.data.joint_vel.torch[:, ids]
        tgt_n = 2.0 * (self.cur_targets[:, ids] - lo) / (hi - lo) - 1.0

        pen_pos, pen_quat, pen_axis = self._pen_state()
        tips = self.hand.data.body_pos_w.torch[:, self.fingertip_ids] - self.scene.env_origins.unsqueeze(1)
        tips_rel = tips - pen_pos.unsqueeze(1)                                           # (N,5,3)
        along = (tips_rel * pen_axis.unsqueeze(1)).sum(-1, keepdim=True)
        along = along.clamp(-0.5 * self.cfg.pen_length, 0.5 * self.cfg.pen_length)
        line_dist = torch.norm(tips_rel - along * pen_axis.unsqueeze(1), dim=-1)          # (N,5)
        contact = torch.norm(self.contact.data.net_forces_w.torch, dim=-1).clamp(max=10.0) / 10.0  # (N,5)

        obs = torch.cat(
            [
                q_n,                                                   # 22
                qd,                                                    # 22
                tgt_n,                                                 # 22
                self.actions,                                          # 22
                10.0 * (pen_pos - self.pen_start_pos),                 # 3
                xyzw_to_wxyz(pen_quat),                                # 4  (w,x,y,z)
                pen_axis,                                              # 3
                self.object.data.root_lin_vel_w.torch,                 # 3
                0.2 * self.object.data.root_ang_vel_w.torch,           # 3
                10.0 * tips_rel.reshape(self.num_envs, -1),            # 15
                10.0 * line_dist,                                      # 5
                contact,                                               # 5
                torch.full((self.num_envs, 1), self.gravity / 9.81, device=self.device),  # 1
            ],
            dim=-1,
        )                                                              # = 130
        return {"policy": obs.clamp(-self.cfg.clip_obs, self.cfg.clip_obs)}

    # ---------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        cfg, ids = self.cfg, self.actuated_ids
        pen_pos, _, pen_axis = self._pen_state()

        omega = self.object.data.root_ang_vel_w.torch
        # Spin reward: heading (yaw) change of the pen's long axis achieved over this policy step.
        # Rolling the pen about its own axis does not change the heading, and back-and-forth motion
        # cancels out under the symmetric clip, so only real net rotation is rewarded.
        yaw = torch.atan2(pen_axis[:, 1], pen_axis[:, 0])
        d = torch.remainder(yaw - self.pen_prev_yaw + math.pi, 2 * math.pi) - math.pi
        heading_rate = d / self.step_dt
        self.spin_angle += d
        self.pen_prev_yaw[:] = yaw
        rot_rew = heading_rate.clamp(-cfg.rotate_clip, cfg.rotate_clip)

        tilt = pen_axis[:, 2].abs()
        pos_err = torch.norm(pen_pos - self.pen_start_pos, dim=-1)
        linvel = torch.norm(self.object.data.root_lin_vel_w.torch, dim=-1)
        pose_diff = ((self.hand.data.joint_pos.torch[:, ids] - self.default_dof_pos[:, ids]) ** 2).sum(-1)
        tau = self.hand.data.applied_torque.torch[:, ids]
        torque = (tau**2).sum(-1)
        work = (tau * self.hand.data.joint_vel.torch[:, ids]).sum(-1).abs()
        act_rate = ((self.actions - self.prev_actions) ** 2).sum(-1)

        reward = (
            cfg.rotate_reward_scale * rot_rew
            + cfg.tilt_penalty_scale * tilt
            + cfg.pos_penalty_scale * pos_err
            + cfg.linvel_penalty_scale * linvel
            + cfg.pose_diff_penalty_scale * pose_diff
            + cfg.torque_penalty_scale * torque
            + cfg.work_penalty_scale * work
            + cfg.action_rate_penalty_scale * act_rate
            + cfg.fall_penalty * self.reset_terminated.float()
        )

        log = self.extras.setdefault("log", {})
        log["metric/heading_rate"] = heading_rate.mean()
        log["rew/rotate"] = rot_rew.mean()
        log["rew/tilt"] = tilt.mean()
        log["rew/pos_err"] = pos_err.mean()
        log["rew/torque"] = torque.mean()
        log["metric/roll_rate"] = (omega * pen_axis).sum(-1).abs().mean()
        log["metric/gravity"] = torch.tensor(self.gravity)
        log["metric/drop_rate_ema"] = torch.tensor(self.drop_rate_ema)
        return reward

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pen_pos, _, pen_axis = self._pen_state()
        fell = pen_pos[:, 2] < self.pen_start_pos[:, 2] - self.cfg.fall_height
        drift = torch.norm(pen_pos - self.pen_start_pos, dim=-1) > self.cfg.max_pen_drift
        tilted = pen_axis[:, 2].abs() > math.sin(self.cfg.max_tilt)
        terminated = fell | drift | tilted
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        rate = terminated.float().mean().item()
        self.drop_rate_ema = 0.99 * self.drop_rate_ema + 0.01 * rate
        if (
            self.cfg.gravity_curriculum
            and self.common_step_counter % 200 == 0
            and self.common_step_counter > 0
            and self.drop_rate_ema < self.cfg.gravity_drop_rate_threshold
            and self.gravity < self.cfg.gravity_max
        ):
            self._set_gravity(min(self.gravity + self.cfg.gravity_step, self.cfg.gravity_max))
            self.drop_rate_ema = 1.0
        return terminated, time_out

    def _set_gravity(self, g: float):
        self.gravity = g
        from isaaclab_physx.physics import PhysxManager  # lazy: needs the running simulator

        PhysxManager.set_gravity((0.0, 0.0, -g))

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        if (self.episode_length_buf[env_ids] > 0).any():
            log = self.extras.setdefault("log", {})
            log["metric/turns_per_episode"] = (self.spin_angle[env_ids] / (2 * math.pi)).mean()
            log["metric/episode_len_s"] = (self.episode_length_buf[env_ids].float() * self.step_dt).mean()

        super()._reset_idx(env_ids)
        k, dev, ids = len(env_ids), self.device, self.actuated_ids

        pen_pose = self.object.data.default_root_pose.torch[env_ids].clone()   # (k,7) pos + quat(xyzw)
        dof_pos = self.default_dof_pos[env_ids].clone()
        if self.grasp_cache is not None:
            g = self.grasp_cache[torch.randint(0, self.grasp_cache.shape[0], (k,), device=dev)]
            dof_pos[:, ids] = g[:, :22]
            pen_pose[:, 0:3] = g[:, 22:25]
            pen_pose[:, 3:7] = g[:, 25:29]
        else:
            noise = self.cfg.reset_joint_noise * (2 * torch.rand((k, len(ids)), device=dev) - 1)
            dof_pos[:, ids] = saturate(dof_pos[:, ids] + noise, self.act_lower[env_ids], self.act_upper[env_ids])
            yaw = self.cfg.reset_yaw_range * (2 * torch.rand(k, device=dev) - 1)
            z = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(k, 1)
            pen_pose[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, z), pen_pose[:, 3:7])
        self.pen_start_pos[env_ids] = pen_pose[:, 0:3]
        pen_pose[:, 0:3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim_index(root_pose=pen_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(root_velocity=torch.zeros((k, 6), device=dev), env_ids=env_ids)

        self.cur_targets[env_ids] = dof_pos
        self.hand.actuators.target_command.set_position_index(value=dof_pos, env_ids=env_ids)
        self.hand.write_joint_position_to_sim_index(position=dof_pos, env_ids=env_ids)
        self.hand.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(dof_pos), env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.rb_forces[env_ids] = 0.0
        self.spin_angle[env_ids] = 0.0
        axis = quat_apply(pen_pose[:, 3:7], self.pen_local_axis[env_ids])
        self.pen_prev_yaw[env_ids] = torch.atan2(axis[:, 1], axis[:, 0])
