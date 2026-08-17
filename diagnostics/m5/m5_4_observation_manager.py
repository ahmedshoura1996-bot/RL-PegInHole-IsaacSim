from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaac_lab  # noqa: F401

from isaac_lab.mdp.observations import peg_hole_relative_position


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"


print("=" * 70)
print("M5.4 OBSERVATION MANAGER VALIDATION")
print("=" * 70)


env = gym.make(
    ENV_NAME,
    num_envs=4,
)

base_env = env.unwrapped

obs, info = env.reset()


print("\n===== POLICY OBSERVATION =====")

policy_obs = obs["policy"]

print("shape:", policy_obs.shape)
print("expected:", (base_env.num_envs, 36))


print("\n===== EXPECTED OBSERVATION LAYOUT =====")

print("joint_pos              : indices [0:9]")
print("joint_vel              : indices [9:18]")
print("object_position        : indices [18:21]")
print("target_object_position : indices [21:28]")
print("actions                : indices [28:36]")


print("\n===== OBJECT POSITION FROM POLICY =====")

object_position_obs = policy_obs[:, 18:21]

print("object_position observation:")
print(object_position_obs)

print("shape:", object_position_obs.shape)


print("\n===== DIRECT PEG-HOLE OBSERVATION =====")

relative_pos = peg_hole_relative_position(base_env)

print("peg_hole_relative_position:")
print(relative_pos)

print("shape:", relative_pos.shape)


print("\n===== CONSISTENCY CHECK =====")

difference = object_position_obs - relative_pos

print("Difference:")
print(difference)

max_error = torch.max(torch.abs(difference)).item()

print("Maximum absolute error:", max_error)


print("\n===== XY CHECK =====")

xy_error = torch.norm(
    relative_pos[:, :2],
    dim=1,
)

print("XY error:", xy_error)

print("Min:", xy_error.min().item())
print("Max:", xy_error.max().item())
print("Mean:", xy_error.mean().item())


print("\n===== PASS CONDITIONS =====")

shape_pass = tuple(policy_obs.shape) == (
    base_env.num_envs,
    36,
)

object_position_shape_pass = tuple(object_position_obs.shape) == (
    base_env.num_envs,
    3,
)

consistency_pass = max_error < 1e-6

xy_centered_pass = bool(
    torch.all(xy_error < 1e-4)
)


print("Policy shape PASS:", shape_pass)
print(
    "Object position shape PASS:",
    object_position_shape_pass,
)
print(
    "Observation/function consistency PASS:",
    consistency_pass,
)
print("XY centered PASS:", xy_centered_pass)


print("\n===== M5.4 FINAL RESULT =====")

all_pass = (
    shape_pass
    and object_position_shape_pass
    and consistency_pass
    and xy_centered_pass
)

print("ALL M5.4 PASS:", all_pass)

print("=" * 70)


env.close()
simulation_app.close()
