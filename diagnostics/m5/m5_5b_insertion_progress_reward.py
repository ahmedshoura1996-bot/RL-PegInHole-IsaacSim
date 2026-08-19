from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab  # noqa: F401
import torch

from isaac_lab.mdp.rewards import peg_insertion_progress


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

# ---------------------------------------------------------------------------
# M5.5B definition
#
# The environment starts with the peg center at Z = 25 mm.
# We define insertion progress relative to this initial state.
#
# Initial peg center Z = 0.025 m
# Target additional insertion = 0.010 m
#
# Therefore:
#   Z = 25 mm -> progress = 0 mm  -> reward = 0.00
#   Z = 22.5 mm -> progress = 2.5 mm -> reward = 0.25
#   Z = 20 mm -> progress = 5 mm -> reward = 0.50
#   Z = 17.5 mm -> progress = 7.5 mm -> reward = 0.75
#   Z = 15 mm -> progress = 10 mm -> reward = 1.00
# ---------------------------------------------------------------------------

INITIAL_PEG_Z = 0.025
TARGET_INSERTION = 0.010

print("=" * 70)
print("M5.5B PEG-IN-HOLE INSERTION PROGRESS REWARD VALIDATION")
print("=" * 70)

env = gym.make(
    ENV_NAME,
    num_envs=4,
)

base_env = env.unwrapped
env.reset()

peg = base_env.scene["object"]

cases = {
    "INITIAL_0MM": 0.000,
    "INSERT_2_5MM": 0.0025,
    "INSERT_5MM": 0.0050,
    "INSERT_7_5MM": 0.0075,
    "INSERT_10MM": 0.0100,
    "RETRACT_2_5MM": -0.0025,
}

results = {}

for name, insertion in cases.items():

    # Positive insertion means moving the peg downward.
    target_z = INITIAL_PEG_Z - insertion

    local_target = torch.tensor(
        [
            0.5,
             0.0,
            target_z,
        ],
        device=base_env.device,
        dtype=torch.float32,
    )

    # Convert local position to world coordinates for all environments.
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

    # Update state without advancing physics.
    # This validates the mathematical reward itself.
    base_env.scene.update(base_env.cfg.sim.dt)

    reward = peg_insertion_progress(
        base_env,
        initial_peg_z=INITIAL_PEG_Z,
        target_insertion=TARGET_INSERTION,
    )

    results[name] = reward.clone()

    actual_z = peg.data.root_pos_w[:, 2] - base_env.scene.env_origins[:, 2]

    actual_progress = INITIAL_PEG_Z - actual_z

    print(f"\n===== {name} =====")

    print("Expected insertion:")
    print(
        torch.full(
            (base_env.num_envs,),
            insertion,
            device=base_env.device,
            dtype=torch.float32,
        )
    )

    print("Peg center Z:")
    print(actual_z)

    print("Actual insertion progress:")
    print(actual_progress)

    print("Reward:")
    print(reward)

    print("Mean:", reward.mean().item())
    print("Min :", reward.min().item())
    print("Max :", reward.max().item())


print("\n" + "=" * 70)
print("M5.5B FINAL CHECK")
print("=" * 70)

initial = results["INITIAL_0MM"]
insert_2_5 = results["INSERT_2_5MM"]
insert_5 = results["INSERT_5MM"]
insert_7_5 = results["INSERT_7_5MM"]
insert_10 = results["INSERT_10MM"]
retract = results["RETRACT_2_5MM"]

print("Initial reward:", initial)
print("2.5 mm reward:", insert_2_5)
print("5 mm reward:", insert_5)
print("7.5 mm reward:", insert_7_5)
print("10 mm reward:", insert_10)
print("Retracted 2.5 mm reward:", retract)


# ---------------------------------------------------------------------------
# PASS 1: Initial state must have zero progress.
# ---------------------------------------------------------------------------

initial_pass = bool(
    torch.allclose(
        initial,
        torch.zeros_like(initial),
        atol=1e-6,
    )
)


# ---------------------------------------------------------------------------
# PASS 2: Expected linear progress values.
# ---------------------------------------------------------------------------

expected_2_5 = torch.full_like(insert_2_5, 0.25)
expected_5 = torch.full_like(insert_5, 0.50)
expected_7_5 = torch.full_like(insert_7_5, 0.75)
expected_10 = torch.full_like(insert_10, 1.00)

linear_pass = bool(
    torch.allclose(insert_2_5, expected_2_5, atol=1e-6)
    and torch.allclose(insert_5, expected_5, atol=1e-6)
    and torch.allclose(insert_7_5, expected_7_5, atol=1e-6)
    and torch.allclose(insert_10, expected_10, atol=1e-6)
)


# ---------------------------------------------------------------------------
# PASS 3: More insertion must always produce a higher reward.
# ---------------------------------------------------------------------------

monotonic_pass = bool(
    torch.all(insert_2_5 > initial)
    and torch.all(insert_5 > insert_2_5)
    and torch.all(insert_7_5 > insert_5)
    and torch.all(insert_10 > insert_7_5)
)


# ---------------------------------------------------------------------------
# PASS 4: Target insertion must saturate at reward = 1.
# ---------------------------------------------------------------------------

target_pass = bool(
    torch.allclose(
        insert_10,
        torch.ones_like(insert_10),
        atol=1e-6,
    )
)


# ---------------------------------------------------------------------------
# PASS 5: Retraction must not receive positive insertion reward.
# ---------------------------------------------------------------------------

retraction_pass = bool(
    torch.all(retract <= 0.0 + 1e-6)
)


print("\nPASS CONDITIONS")
print("Initial reward = 0:", initial_pass)
print("Expected insertion values:", linear_pass)
print("Monotonic insertion progress:", monotonic_pass)
print("10 mm target reward = 1:", target_pass)
print("Retraction gives no positive reward:", retraction_pass)

all_pass = (
    initial_pass
    and linear_pass
    and monotonic_pass
    and target_pass
    and retraction_pass
)

print("\nM5.5B ALL PASS:", all_pass)

env.close()
simulation_app.close()
