import torch

from isaaclab.app import AppLauncher

# Start Isaac Sim FIRST
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab  # registers the Peg-in-Hole environment

ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

print("\n" + "=" * 70)
print("PEG-IN-HOLE ENVIRONMENT STEP TEST")
print("=" * 70)

print("\n[1] Creating environment...")
env = gym.make(ENV_NAME)
print("[OK] Environment created")
print("Environment:", env)

print("\n[2] Resetting environment...")
obs, info = env.reset()
print("[OK] Reset successful")

print("\nObservation type:")
print(type(obs))

print("\nObservation:")
print(obs)

if isinstance(obs, dict):
    for key, value in obs.items():
        print(f"\nObservation group: {key}")
        print("Shape:", value.shape)
        print("Device:", value.device)
        print("Min:", value.min().item())
        print("Max:", value.max().item())

print("\n[3] Checking action space...")
print("Action space:")
print(env.action_space)

print("\n[4] Generating zero action...")

action = torch.zeros(
    (4096, 8),
    device="cuda:0",
    dtype=torch.float32,
)

print("Action:")
print(action)

print("\n[5] Performing one environment step...")

obs, reward, terminated, truncated, info = env.step(action)

print("[OK] STEP SUCCESS")

print("\nReward:")
print(reward)

print("\nTerminated:")
print(terminated)

print("\nTruncated:")
print(truncated)

print("\n[6] Performing 10 more steps...")

for i in range(10):

    action = torch.zeros(
        (4096, 8),
        device="cuda:0",
        dtype=torch.float32,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {i + 2:02d} | "
        f"Reward Mean = {reward.mean().item(): .6f} | "
        f"Terminated = {terminated.any().item()} | "
        f"Truncated = {truncated.any().item()}"
    )

print("\n[7] Closing environment...")

env.close()

print("[OK] Environment closed")

simulation_app.close()

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
