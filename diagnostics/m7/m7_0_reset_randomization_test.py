from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaac_lab

from isaaclab_tasks.utils import parse_env_cfg


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"
NUM_ENVS = 4096
NUM_RESETS = 5

print("==============================================")
print("M7.0 RESET RANDOMIZATION VALIDATION")
print("==============================================")

env_cfg = parse_env_cfg(
    ENV_NAME,
    device="cuda:0",
    num_envs=NUM_ENVS,
    use_fabric=True,
)

env = gym.make(ENV_NAME, cfg=env_cfg)
manager_env = env.unwrapped

print("Environment creation: PASS")

object_asset = manager_env.scene["object"]

for reset_id in range(NUM_RESETS):

    obs, _ = env.reset()

    object_pos = object_asset.data.root_pos_w
    env_origins = manager_env.scene.env_origins

    local_pos = object_pos - env_origins

    # Hole center in environment-local coordinates.
    hole_x = 0.5
    hole_y = 0.0

    # Measure PEG displacement relative to the hole center.
    x_offset = local_pos[:, 0] - hole_x
    y_offset = local_pos[:, 1] - hole_y

    x_mm = x_offset * 1000.0
    y_mm = y_offset * 1000.0

    print()
    print(f"Reset {reset_id + 1}/{NUM_RESETS}")
    print("----------------------------------------------")
    print(f"X range : [{x_mm.min().item():+.4f}, {x_mm.max().item():+.4f}] mm")
    print(f"Y range : [{y_mm.min().item():+.4f}, {y_mm.max().item():+.4f}] mm")
    print(f"X mean  : {x_mm.mean().item():+.4f} mm")
    print(f"Y mean  : {y_mm.mean().item():+.4f} mm")
    print(f"X std   : {x_mm.std().item():.4f} mm")
    print(f"Y std   : {y_mm.std().item():.4f} mm")

    valid_x = (x_offset >= -0.001) & (x_offset <= 0.001)
    valid_y = (y_offset >= -0.001) & (y_offset <= 0.001)

    valid = valid_x & valid_y

    print(f"Within ±1 mm: {valid.sum().item()} / {NUM_ENVS}")

    if not torch.all(valid):
        raise RuntimeError("M7.0 FAILED: reset positions exceeded ±1 mm.")

print()
print("==============================================")
print("M7.0 RESULTS")
print("==============================================")
print("Reset randomization range: ±1.0 mm")
print(f"Number of environments:    {NUM_ENVS}")
print(f"Number of resets:          {NUM_RESETS}")
print("Range validation:          PASS")
print("==============================================")

env.close()
simulation_app.close()
