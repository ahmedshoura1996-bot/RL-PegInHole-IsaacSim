from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab  # noqa: F401
import torch


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"


print("=" * 70)
print("M5.2 OBSERVATION RUNTIME VALIDATION")
print("=" * 70)

env = gym.make(
    ENV_NAME,
    num_envs=4,
)

base_env = env.unwrapped

obs, info = env.reset()

print("\n===== ENVIRONMENT =====")
print("num_envs:", base_env.num_envs)
print("device:", base_env.device)

print("\n===== OBSERVATION FUNCTIONS =====")

from isaac_lab.mdp.observations import (
    peg_position,
    peg_hole_relative_position,
)

peg_pos = peg_position(base_env)
relative_pos = peg_hole_relative_position(base_env)

print("peg_position:")
print(peg_pos)
print("shape:", peg_pos.shape)

print("\npeg_hole_relative_position:")
print(relative_pos)
print("shape:", relative_pos.shape)

print("\n===== XY ERROR =====")

xy_error = torch.norm(relative_pos[:, :2], dim=1)

print("XY error:", xy_error)
print("Min:", xy_error.min().item())
print("Max:", xy_error.max().item())
print("Mean:", xy_error.mean().item())

print("\n===== Z ERROR =====")

z_error = relative_pos[:, 2]

print("Z relative:", z_error)
print("Min:", z_error.min().item())
print("Max:", z_error.max().item())
print("Mean:", z_error.mean().item())

print("\n===== PASS CONDITIONS =====")

print(
    "peg_position shape PASS:",
    tuple(peg_pos.shape) == (base_env.num_envs, 3),
)

print(
    "relative position shape PASS:",
    tuple(relative_pos.shape) == (base_env.num_envs, 3),
)

print(
    "XY centered PASS:",
    bool(torch.all(xy_error < 1e-4)),
)

print("=" * 70)

env.close()
simulation_app.close()
