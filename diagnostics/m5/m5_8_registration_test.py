from isaaclab.app import AppLauncher

app_launcher = AppLauncher({
    "headless": True,
})
simulation_app = app_launcher.app

print("=== M5.8 REGISTRATION TEST ===")

import gymnasium as gym
import isaac_lab

ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

spec = gym.spec(ENV_NAME)

print("Environment:", spec.id)

entry_point = spec.kwargs.get("rsl_rl_cfg_entry_point")

print("PPO entry point:", entry_point)

assert entry_point is not None, "FAIL: rsl_rl_cfg_entry_point is missing"

expected = "isaac_lab.agents.rsl_rl_ppo_cfg:PegInHolePPORunnerCfg"

assert entry_point == expected, (
    f"FAIL: unexpected PPO entry point: {entry_point}"
)

module_name, class_name = entry_point.split(":")

module = __import__(module_name, fromlist=[class_name])
cfg_class = getattr(module, class_name)

cfg = cfg_class()

print("PPO config class:", cfg_class.__name__)
print("experiment_name:", cfg.experiment_name)
print("num_steps_per_env:", cfg.num_steps_per_env)
print("max_iterations:", cfg.max_iterations)
print("learning_rate:", cfg.algorithm.learning_rate)
print("gamma:", cfg.algorithm.gamma)

print("=== M5.8 PASS ===")

simulation_app.close()
