from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab  # noqa: F401
import torch

from isaac_lab.mdp.observations import peg_hole_relative_position
from isaac_lab.mdp.rewards import (
    peg_hole_xy_alignment,
    peg_insertion_progress,
)


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

XY_WEIGHT = 0.5
INSERTION_WEIGHT = 0.5

INITIAL_PEG_Z = 0.025
TARGET_INSERTION = 0.010


print("=" * 70)
print("M5.5C PEG-IN-HOLE COMBINED REWARD VALIDATION")
print("=" * 70)


env = gym.make(
    ENV_NAME,
    num_envs=4,
)

base_env = env.unwrapped
env.reset()

peg = base_env.scene["object"]

hole_x = 0.5


cases = {
    "CENTER_0MM": (0.0, 0.0, 0.000),
    "CENTER_5MM_INSERTION": (0.0, 0.0, 0.005),
    "CENTER_10MM_INSERTION": (0.0, 0.0, 0.010),
    "OFFSET_5MM_0MM": (0.005, 0.0, 0.000),
    "OFFSET_5MM_5MM": (0.005, 0.0, 0.005),
    "OFFSET_5MM_10MM": (0.005, 0.0, 0.010),
    "DIAGONAL_5MM_10MM": (0.005, 0.005, 0.010),
}


results = {}


for name, (dx, dy, insertion) in cases.items():

    local_target = torch.tensor(
        [
            hole_x + dx,
            dy,
            INITIAL_PEG_Z - insertion,
        ],
        device=base_env.device,
        dtype=torch.float32,
    )

    target_world = (
        base_env.scene.env_origins
        + local_target.unsqueeze(0)
    )

    root_pose = torch.cat(
        [
            target_world,
            peg.data.root_quat_w.clone(),
        ],
        dim=1,
    )

    root_velocity = torch.zeros(
        (base_env.num_envs, 6),
        device=base_env.device,
        dtype=torch.float32,
    )

    peg.write_root_pose_to_sim(root_pose)
    peg.write_root_velocity_to_sim(root_velocity)

    # Validate the mathematical reward without advancing physics.
    base_env.scene.update(base_env.cfg.sim.dt)

    relative = peg_hole_relative_position(base_env)

    xy_reward = peg_hole_xy_alignment(base_env)

    insertion_reward = peg_insertion_progress(
        base_env,
        initial_peg_z=INITIAL_PEG_Z,
        target_insertion=TARGET_INSERTION,
    )

    combined_reward = (
        XY_WEIGHT * xy_reward
        + INSERTION_WEIGHT * insertion_reward
    )

    results[name] = combined_reward.clone()

    print(f"\n===== {name} =====")

    print("Expected XY offset:")
    print(
        torch.tensor(
            [dx, dy],
            device=base_env.device,
            dtype=torch.float32,
        )
    )

    print("Expected insertion:")
    print(
        torch.full(
            (base_env.num_envs,),
            insertion,
            device=base_env.device,
            dtype=torch.float32,
        )
    )

    print("Relative position:")
    print(relative)

    print("XY reward:")
    print(xy_reward)

    print("Insertion reward:")
    print(insertion_reward)

    print("Combined reward:")
    print(combined_reward)

    print("Mean:", combined_reward.mean().item())


print("\n" + "=" * 70)
print("M5.5C FINAL CHECK")
print("=" * 70)


center_0 = results["CENTER_0MM"]
center_5 = results["CENTER_5MM_INSERTION"]
center_10 = results["CENTER_10MM_INSERTION"]

offset_0 = results["OFFSET_5MM_0MM"]
offset_5 = results["OFFSET_5MM_5MM"]
offset_10 = results["OFFSET_5MM_10MM"]

diagonal_10 = results["DIAGONAL_5MM_10MM"]


print("Center 0 mm:", center_0)
print("Center 5 mm:", center_5)
print("Center 10 mm:", center_10)
print("Offset 5 mm / 0 mm:", offset_0)
print("Offset 5 mm / 5 mm:", offset_5)
print("Offset 5 mm / 10 mm:", offset_10)
print("Diagonal 5/5 mm / 10 mm:", diagonal_10)


# PASS 1:
# Perfect alignment + no insertion gives 0.5.
center_zero_pass = bool(
    torch.allclose(
        center_0,
        torch.full_like(center_0, 0.5),
        atol=1e-6,
    )
)


# PASS 2:
# Perfect alignment + 10 mm insertion gives 1.0.
center_target_pass = bool(
    torch.allclose(
        center_10,
        torch.ones_like(center_10),
   atol=1e-6,
    )
)


# PASS 3:
# At perfect alignment, insertion progress increases reward.
monotonic_pass = bool(
    torch.all(center_0 < center_5)
    and torch.all(center_5 < center_10)
)


# PASS 4:
# At the same insertion depth, better XY alignment gives higher reward.
alignment_pass = bool(
    torch.all(center_10 > offset_10)
)


# PASS 5:
# Diagonal XY error is larger than single-axis error.
diagonal_pass = bool(
    torch.all(diagonal_10 < offset_10)
)


print("\nPASS CONDITIONS")
print("Center 0 mm = 0.5:", center_zero_pass)
print("Center 10 mm = 1:", center_target_pass)
print("Insertion monotonicity:", monotonic_pass)
print("Alignment improves reward:", alignment_pass)
print("Diagonal penalty:", diagonal_pass)


all_pass = (
    center_zero_pass
    and center_target_pass
    and monotonic_pass
    and alignment_pass
    and diagonal_pass
)


print("\nM5.5C ALL PASS:", all_pass)


env.close()
simulation_app.close()
