from isaaclab.app import AppLauncher

app_launcher = AppLauncher({
    "headless": True,
})
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaac_lab
from isaac_lab.mdp.metrics import peg_hole_success


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"


print("=== M6.1 ENVIRONMENT SUCCESS METRIC TEST ===")
print("Environment:", ENV_NAME)
print("Metric:", "peg_hole_success")


env = gym.make(
    ENV_NAME,
    cfg=None,
)

manager_env = env.unwrapped

print("Environment creation: PASS")
print("Number of environments:", manager_env.num_envs)

obs, info = env.reset()

print("Environment reset: PASS")

success = peg_hole_success(manager_env)

print("Metric execution: PASS")
print("Metric shape:", tuple(success.shape))
print("Metric dtype:", success.dtype)
print("Metric device:", success.device)

assert success.shape == (manager_env.num_envs,)
assert success.dtype == torch.float32
assert torch.all((success == 0.0) | (success == 1.0))

print("Output shape: PASS")
print("Output dtype: PASS")
print("Binary output: PASS")

print("Initial success count:", int(success.sum().item()))
print("Initial success rate:", float(success.mean().item()))

env.close()
simulation_app.close()

print("=== M6.1 ENVIRONMENT VALIDATION: PASS ===")
