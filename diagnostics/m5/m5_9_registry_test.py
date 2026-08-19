from isaaclab.app import AppLauncher

app_launcher = AppLauncher({
    "headless": True,
})
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab

ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

spec = gym.spec(ENV_NAME)

print("=== M5.9 REGISTRY TEST ===")
print("Environment:", spec.id)

env_cfg = spec.kwargs.get("env_cfg_entry_point")
ppo_cfg = spec.kwargs.get("rsl_rl_cfg_entry_point")

print("ENV CFG:", env_cfg)
print("PPO CFG:", ppo_cfg)

assert env_cfg == "isaac_lab.peg_in_hole_env_cfg:PegInHoleEnvCfg"
assert ppo_cfg == "isaac_lab.agents.rsl_rl_ppo_cfg:PegInHolePPORunnerCfg"

print("=== M5.9 REGISTRY PASS ===")

simulation_app.close()
