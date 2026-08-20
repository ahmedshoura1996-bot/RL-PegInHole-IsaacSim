from isaaclab.app import AppLauncher

app_launcher = AppLauncher({
    "headless": True,
})
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch

import isaac_lab

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaac_lab.mdp.metrics import peg_hole_success


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

CHECKPOINT = (
    "/workspace/projects/RL-PegInHole-IsaacSim/"
    "logs/rsl_rl/peg_in_hole/2026-08-19_13-34-31/model_9.pt"
)

NUM_ENVS = 4096
NUM_STEPS = 250


print("==============================================")
print("M6.2 BASELINE POLICY EVALUATION")
print("==============================================")
print("Environment:", ENV_NAME)
print("Checkpoint:", CHECKPOINT)
print("Number of environments:", NUM_ENVS)
print("Evaluation steps:", NUM_STEPS)


assert os.path.isfile(CHECKPOINT), (
    f"Checkpoint not found: {CHECKPOINT}"
)

print("Checkpoint exists: PASS")


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

env_cfg = parse_env_cfg(
    ENV_NAME,
    device="cuda:0",
    num_envs=NUM_ENVS,
    use_fabric=True,
)

env = gym.make(
    ENV_NAME,
    cfg=env_cfg,
)

manager_env = env.unwrapped

print("Environment creation: PASS")
print("Number of environments:", manager_env.num_envs)


# ------------------------------------------------------------
# Reset
# ------------------------------------------------------------

obs, info = env.reset()

print("Environment reset: PASS")


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location=manager_env.device,
)

print("Checkpoint load: PASS")
print("Checkpoint keys:", list(checkpoint.keys()))


# ------------------------------------------------------------
# Locate policy
# ------------------------------------------------------------

if "model_state_dict" in checkpoint:
    print("Checkpoint contains model_state_dict: PASS")
else:
    print("WARNING: model_state_dict not found directly.")

print("Observation dimension:", manager_env.observation_manager.group_obs_dim)
print("Action dimension:", manager_env.action_manager.total_action_dim)


# ------------------------------------------------------------
# Initial metric
# ------------------------------------------------------------

success = peg_hole_success(manager_env)

print("Initial success count:", int(success.sum().item()))
print("Initial success rate:", float(success.mean().item()))


# ------------------------------------------------------------
# Evaluation placeholder
# ------------------------------------------------------------

print()
print("==============================================")
print("M6.2 CHECKPOINT INSPECTION COMPLETE")
print("==============================================")
print()
print("Next step:")
print("Connect the RSL-RL policy runner to this")
print("environment and execute inference evaluation.")
print()


env.close()
simulation_app.close()

print("=== M6.2.1 INFRASTRUCTURE CHECK: PASS ===")
