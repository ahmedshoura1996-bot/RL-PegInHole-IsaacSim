from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaac_lab

from isaac_lab.agents.rsl_rl_ppo_cfg import PegInHolePPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaac_lab.mdp.observations import peg_hole_relative_position
from isaac_lab.mdp.rewards import (
    peg_hole_xy_alignment,
    peg_insertion_progress,
)


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

CHECKPOINT = (
    "/workspace/projects/RL-PegInHole-IsaacSim/"
    "logs/rsl_rl/peg_in_hole/2026-08-19_13-34-31/model_9.pt"
)

NUM_ENVS = 4096
NUM_STEPS = 250

XY_TOLERANCE = 0.0005
INSERTION_TARGET = 0.010
INITIAL_PEG_Z = 0.025


print("==============================================")
print("M6.3 BASELINE FAILURE ANALYSIS")
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

env = RslRlVecEnvWrapper(env)

obs, _ = env.reset()

agent_cfg = PegInHolePPORunnerCfg()

runner = OnPolicyRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=None,
    device="cuda:0",
)

runner.load(CHECKPOINT)

policy = runner.get_inference_policy(device="cuda:0")

print("Checkpoint loaded: PASS")
print("Inference policy extracted: PASS")


# ------------------------------------------------------------
# Initial state
# ------------------------------------------------------------

peg = manager_env.scene["object"]

initial_z = peg.data.root_pos_w[:, 2].clone()

best_xy = torch.full(
    (NUM_ENVS,),
    float("inf"),
    device=manager_env.device,
)

best_insertion = torch.zeros(
    NUM_ENVS,
    device=manager_env.device,
)

max_xy_reward = torch.full(
    (NUM_ENVS,),
    -float("inf"),
    device=manager_env.device,
)

max_insertion_reward = torch.full(
    (NUM_ENVS,),
    -float("inf"),
    device=manager_env.device,
)

max_total_reward = torch.full(
    (NUM_ENVS,),
    -float("inf"),
    device=manager_env.device,
)

action_sum = 0.0

print()
print("Running failure analysis...")


for step in range(NUM_STEPS):

    with torch.inference_mode():
        actions = policy(obs)

    action_sum += actions.abs().mean().item()

    obs, rewards, dones, extras = env.step(actions)

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    relative_pos = peg_hole_relative_position(
        manager_env
    )

    xy_error = torch.norm(
        relative_pos[:, :2],
        dim=1,
    )

    peg_z = (
        peg.data.root_pos_w[:, 2]
 - manager_env.scene.env_origins[:, 2]
    )

    insertion = INITIAL_PEG_Z - peg_z

    # --------------------------------------------------------
    # Rewards
    # --------------------------------------------------------

    xy_reward = peg_hole_xy_alignment(
        manager_env,
        std=0.01,
    )

    insertion_reward = peg_insertion_progress(
        manager_env,
        initial_peg_z=INITIAL_PEG_Z,
        target_insertion=INSERTION_TARGET,
    )

    total_reward = (
        0.5 * xy_reward
        + 0.5 * insertion_reward
    )

    # --------------------------------------------------------
    # Track extrema
    # --------------------------------------------------------

    best_xy = torch.minimum(
        best_xy,
        xy_error,
    )

    best_insertion = torch.maximum(
        best_insertion,
        insertion,
    )

    max_xy_reward = torch.maximum(
        max_xy_reward,
        xy_reward,
    )

    max_insertion_reward = torch.maximum(
        max_insertion_reward,
     insertion_reward,
    )

    max_total_reward = torch.maximum(
        max_total_reward,
        total_reward,
    )

    if (step + 1) % 25 == 0:

        print(
            f"Step {step + 1:3d} | "
            f"XY={xy_error.mean().item():.6f} m | "
            f"Insert={insertion.mean().item():.6f} m | "
            f"XY_R={xy_reward.mean().item():.4f} | "
            f"INS_R={insertion_reward.mean().item():.4f} | "
            f"Total_R={total_reward.mean().item():.4f}"
        )


# ------------------------------------------------------------
# Final analysis
# ------------------------------------------------------------

final_relative = peg_hole_relative_position(manager_env)

final_xy = torch.norm(
    final_relative[:, :2],
    dim=1,
)

final_z = (
    peg.data.root_pos_w[:, 2]
    - manager_env.scene.env_origins[:, 2]
)

final_insertion = INITIAL_PEG_Z - final_z

xy_aligned = best_xy <= XY_TOLERANCE
inserted = best_insertion >= INSERTION_TARGET
successful = xy_aligned & inserted

# Contact / near-hole classification
near_hole_1mm = best_xy <= 0.001
near_hole_5mm = best_xy <= 0.005

insertion_1mm = best_insertion >= 0.001
insertion_3mm = best_insertion >= 0.003
insertion_5mm = best_insertion >= 0.005


print()
print("==============================================")
print("M6.3 RESULTS")
print("==============================================")

print()
print("---- GEOMETRY ----")

print(
    f"Final mean XY error:       "
    f"{final_xy.mean().item():.6f} m"
)

print(
    f"Best mean XY error:        "
    f"{best_xy.mean().item():.6f} m"
)

print(
    f"Final mean insertion:      "
    f"{final_insertion.mean().item():.6f} m"
)

print(
    f"Best mean insertion:       "
    f"{best_insertion.mean().item():.6f} m"
)

print()
print("---- REWARD ----")

print(
    f"Max mean XY reward:        "
    f"{max_xy_reward.mean().item():.6f}"
)

print(
    f"Max mean insertion reward: "
    f"{max_insertion_reward.mean().item():.6f}"
)

print(
    f"Max mean combined reward:  "
    f"Max mean combined reward:  "
    f"{max_total_reward.mean().item():.6f}"
)

print()
print("---- THRESHOLDS ----")

print(
    f"XY aligned <= 0.5 mm:      "
    f"{int(xy_aligned.sum())} / {NUM_ENVS}"
)

print(
    f"Within 1 mm XY:            "
    f"{int(near_hole_1mm.sum())} / {NUM_ENVS}"
)

print(
    f"Within 5 mm XY:             "
    f"{int(near_hole_5mm.sum())} / {NUM_ENVS}"
)

print(
    f"Insertion >= 1 mm:         "
    f"{int(insertion_1mm.sum())} / {NUM_ENVS}"
)

print(
    f"Insertion >= 3 mm:         "
    f"{int(insertion_3mm.sum())} / {NUM_ENVS}"
)

print(
    f"Insertion >= 5 mm:         "
    f"{int(insertion_5mm.sum())} / {NUM_ENVS}"
)

print(
    f"Insertion >= 10 mm:        "
    f"{int(inserted.sum())} / {NUM_ENVS}"
)

print()
print("---- FINAL ----")

print(
    f"Successful environments:   "
    f"{int(successful.sum())} / {NUM_ENVS}"
)

print(
    f"Success rate:              "
    f"{successful.float().mean().item():.6f}"
)

print(
    f"Mean action magnitude:     "
    f"{action_sum / NUM_STEPS:.6f}"
)

print()
print("==============================================")
print("M6.3 FAILURE CLASSIFICATION")
print("==============================================")

if xy_aligned.all() and not inserted.any():
    print(
        "CLASSIFICATION: "
        "XY ALIGNMENT SUCCESS / INSERTION FAILURE"
    )
elif inserted.any():
    print(
        "CLASSIFICATION: "
        "PARTIAL INSERTION SUCCESS"
    )
else:
    print(
        "CLASSIFICATION: "
        "GENERAL POLICY FAILURE"
    )

print()
print("=== M6.3 DIAGNOSTICS COMPLETE ===")


env.close()
simulation_app.close()
