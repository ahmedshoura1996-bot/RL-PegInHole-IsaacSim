from __future__ import annotations

import os

from isaaclab.app import AppLauncher

# ============================================================
# Isaac Sim
# ============================================================

app_launcher = AppLauncher({
    "headless": True,
})
simulation_app = app_launcher.app

# ============================================================
# Imports after Isaac Sim initialization
# ============================================================

import gymnasium as gym
import torch
import yaml

import isaac_lab  # noqa: F401

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

from isaac_lab.mdp.metrics import peg_hole_success


# ============================================================
# Configuration
# ============================================================

ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

CHECKPOINT = (
    "/workspace/projects/RL-PegInHole-IsaacSim/"
    "logs/rsl_rl/peg_in_hole/2026-08-19_13-34-31/model_9.pt"
)

AGENT_CONFIG = (
    "/workspace/projects/RL-PegInHole-IsaacSim/"
    "logs/rsl_rl/peg_in_hole/2026-08-19_13-34-31/"
    "params/agent.yaml"
)

NUM_ENVS = 4096
NUM_STEPS = 250


print("==============================================")
print(" M6.2.2 RSL-RL POLICY EVALUATION")
print("==============================================")
print("Environment:", ENV_NAME)
print("Checkpoint:", CHECKPOINT)
print("Agent config:", AGENT_CONFIG)
print("Number of environments:", NUM_ENVS)
print("Evaluation steps:", NUM_STEPS)
print("==============================================")


# ============================================================
# Validate files
# ============================================================

assert os.path.isfile(CHECKPOINT), (
     f"Checkpoint not found: {CHECKPOINT}"
)

assert os.path.isfile(AGENT_CONFIG), (
    f"Agent config not found: {AGENT_CONFIG}"
)

print("Checkpoint exists: PASS")
print("Agent config exists: PASS")


# ============================================================
# Load agent configuration
# ============================================================

with open(AGENT_CONFIG, "r") as f:
    agent_cfg_dict = yaml.safe_load(f)

print("Agent configuration loaded: PASS")

print("Runner class:")
print(
    agent_cfg_dict.get("class_name")
)


# ============================================================
# Environment configuration
# ============================================================

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

print("Environment creation: PASS")

manager_env = env.unwrapped

print("Number of environments:", manager_env.num_envs)
print(
    "Observation dimension:",
  manager_env.observation_manager.group_obs_dim,
)
print(
    "Action dimension:",
    manager_env.action_manager.total_action_dim,
)


# ============================================================
# RSL-RL vectorized wrapper
# ============================================================

env = RslRlVecEnvWrapper(
    env,
    clip_actions=agent_cfg_dict.get(
        "clip_actions",
        None,
    ),
)

print("RSL-RL wrapper creation: PASS")


# ============================================================
# Create RSL-RL runner
# ============================================================

runner_class = agent_cfg_dict.get(
    "class_name",
    "OnPolicyRunner",
)

if runner_class == "OnPolicyRunner":

    runner = OnPolicyRunner(
        env,
        agent_cfg_dict,
        log_dir=None,
        device="cuda:0",
    )

elif runner_class == "DistillationRunner":

    runner = DistillationRunner(
        env,
        agent_cfg_dict,
        log_dir=None,
        device="cuda:0",
    )

else:
     raise RuntimeError(
        f"Unsupported RSL-RL runner: {runner_class}"
    )

print("RSL-RL runner creation: PASS")
print("Runner:", runner_class)


# ============================================================
# Load trained checkpoint
# ============================================================

runner.load(CHECKPOINT)

print("Checkpoint loaded into RSL-RL runner: PASS")


# ============================================================
# Inference policy
# ============================================================

policy = runner.get_inference_policy(
    device="cuda:0"
)

print("Inference policy extraction: PASS")


# ============================================================
# Reset environment
# ============================================================

obs, info = env.reset()

print("Environment reset: PASS")


# ============================================================
# Initial success
# ============================================================

initial_success = peg_hole_success(
    manager_env
)

print(
    "Initial success count:",
    int(initial_success.sum().item()),
)

print(
    "Initial success rate:",
    float(initial_success.mean().item()),
)


# ============================================================
# Policy evaluation
# ============================================================

success_count = torch.zeros(
    manager_env.num_envs,
    dtype=torch.bool,
    device=manager_env.device,
)

episodes_completed = torch.zeros(
    manager_env.num_envs,
    dtype=torch.bool,
    device=manager_env.device,
)


print()
print("==============================================")
print(" Running policy inference")
print("==============================================")


for step in range(NUM_STEPS):

    with torch.inference_mode():

        actions = policy(obs)

    obs, rewards, dones, extras = env.step(actions)

    current_success = (
        peg_hole_success(manager_env) > 0.5
    )

    success_count |= current_success

    episodes_completed |= dones.bool()

    if (step + 1) % 50 == 0:

        current_rate = (
            success_count.float().mean().item()
        )

        print(
            f"Step {step + 1:4d} | "
            f"success rate so far: "
            f"{current_rate:.6f}"
        )


# ============================================================
# Final evaluation
# ============================================================

final_success_count = int(
    success_count.sum().item()
)

final_success_rate = float(
    success_count.float().mean().item()
)

completed_count = int(
    episodes_completed.sum().item()
)

completed_rate = float(
    episodes_completed.float().mean().item()
)


print()
print("==============================================")
print(" M6.2.2 EVALUATION RESULT")
print("==============================================")

print(
    "Successful environments:",
    final_success_count,
    "/",
     manager_env.num_envs,
)

print(
    "Success rate:",
    final_success_rate,
)

print(
    "Environments with completed episodes:",
    completed_count,
    "/",
    manager_env.num_envs,
)

print(
    "Episode completion rate:",
    completed_rate,
)

print("Evaluation steps:", NUM_STEPS)


# ============================================================
# Cleanup
# ============================================================

env.close()
simulation_app.close()

print()
print("=== M6.2.2 POLICY INFERENCE: PASS ===")
