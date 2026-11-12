import gymnasium as gym

from . import agents

_AGENTS = {
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PenSpinPPORunnerCfg",
    "default_agent": "rsl_rl",
}

for task_id, cfg_name, env_cls in [
    ("PenSpin-Sharpa", "pen_spin_env_cfg:PenSpinEnvCfg", "pen_spin_env:PenSpinEnv"),
    ("PenSpin-Sharpa-Play", "pen_spin_env_cfg:PenSpinEnvCfg_PLAY", "pen_spin_env:PenSpinEnv"),
    ("PenSpin-Sharpa-Grasp", "grasp_env:PenGraspEnvCfg", "grasp_env:PenGraspEnv"),
]:
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.{env_cls}",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.{cfg_name}", **_AGENTS},
    )
