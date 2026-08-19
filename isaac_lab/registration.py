import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnvCfg


def make_peg_in_hole_env(**kwargs):
    from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg

    # Gymnasium passes the environment configuration through gym.make(..., cfg=...).
    # Reuse it when provided instead of creating a second configuration.
    cfg = kwargs.pop("cfg", None)

    # Remove registry metadata before forwarding kwargs to ManagerBasedRLEnv.
    kwargs.pop("env_cfg_entry_point", None)
    kwargs.pop("rsl_rl_cfg_entry_point", None)

    if cfg is None:
        cfg = PegInHoleEnvCfg()

    if "num_envs" in kwargs:
        cfg.scene.num_envs = kwargs.pop("num_envs")

    return ManagerBasedRLEnv(cfg=cfg, **kwargs)


gym.register(
    id="Isaac-PegInHole-Franka-IK-Abs-v0",
    entry_point=make_peg_in_hole_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_lab.peg_in_hole_env_cfg:PegInHoleEnvCfg",
        "rsl_rl_cfg_entry_point": "isaac_lab.agents.rsl_rl_ppo_cfg:PegInHolePPORunnerCfg",
    },
)
