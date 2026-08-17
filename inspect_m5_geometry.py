import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaac_lab

print("=" * 70)
print("M5 GEOMETRY INSPECTION")
print("=" * 70)

env = gym.make(
    ENV_NAME,
    num_envs=4,
)

env.reset()

base_env = env.unwrapped

for name in [
    "object",
    "fixture_left",
    "fixture_right",
    "fixture_front",
    "fixture_back",
]:
    print(f"\n[{name}]")

    asset = base_env.scene[name]

    print("Root position:")
    print(asset.data.root_pos_w[:4])

    print("Root quaternion:")
    print(asset.data.root_quat_w[:4])

print("\n" + "=" * 70)
print("ROBOT")
print("=" * 70)

robot = base_env.scene["robot"]

print("Robot root:")
print(robot.data.root_pos_w[:4])

print("Joint positions:")
print(robot.data.joint_pos[:4])

print("\n" + "=" * 70)
print("OBSERVATION")
print("=" * 70)

obs, _ = env.reset()

print("Observation keys:", obs.keys())

for key, value in obs.items():
    print(
        f"{key}: shape={value.shape}, "
        f"min={value.min().item():.6f}, "
        f"max={value.max().item():.6f}"
    )

print("\nClosing...")

env.close()
simulation_app.close()
