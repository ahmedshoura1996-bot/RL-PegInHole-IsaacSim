from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch
import isaac_lab

from isaac_lab.agents.rsl_rl_ppo_cfg import PegInHolePPORunnerCfg

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaac_lab.mdp.observations import peg_hole_relative_position


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"
CHECKPOINT = "/workspace/projects/RL-PegInHole-IsaacSim/logs/rsl_rl/peg_in_hole/2026-08-20_13-23-13_m6_5_dense_insertion/model_99.pt"

NUM_ENVS = 4096
NUM_STEPS = 250


print("==============================================")
print("M6.2.3 POLICY BEHAVIOR DIAGNOSTICS")
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

# Build runner using the actual environment dimensions.
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

initial_rel = peg_hole_relative_position(manager_env)
initial_xy = torch.norm(initial_rel[:, :2], dim=1)

min_xy = initial_xy.clone()

initial_z = manager_env.scene["object"].data.root_pos_w[:, 2].clone()
max_insertion = torch.zeros(NUM_ENVS, device=manager_env.device)

action_sum = 0.0

print()
print("Running diagnostics...")

for step in range(NUM_STEPS):

    with torch.inference_mode():
        actions = policy(obs)

    action_sum += actions.abs().mean().item()

    obs, rewards, dones, extras = env.step(actions)

    rel = peg_hole_relative_position(manager_env)
    xy_error = torch.norm(rel[:, :2], dim=1)

    min_xy = torch.minimum(min_xy, xy_error)

    current_z = manager_env.scene["object"].data.root_pos_w[:, 2]
    insertion = initial_z - current_z

    max_insertion = torch.maximum(max_insertion, insertion)

    if (step + 1) % 50 == 0:
        print(
            f"Step {step+1:3d} | "
            f"mean XY: {xy_error.mean().item():.6f} | "
            f"best XY: {min_xy.mean().item():.6f} | "
            f"mean insertion: {insertion.mean().item():.6f}"
        )

final_rel = peg_hole_relative_position(manager_env)
final_xy = torch.norm(final_rel[:, :2], dim=1)

final_z = manager_env.scene["object"].data.root_pos_w[:, 2]
final_insertion = initial_z - final_z

xy_tol = 0.0005
insertion_target = 0.010

xy_aligned = min_xy <= xy_tol
inserted = max_insertion >= insertion_target
success = xy_aligned & inserted

print()
print("==============================================")
print("M6.2.3 RESULTS")
print("==============================================")

print(f"Final mean XY error:       {final_xy.mean().item():.6f} m")
print(f"Best mean XY error:        {min_xy.mean().item():.6f} m")
print(f"Final mean insertion:      {final_insertion.mean().item():.6f} m")
print(f"Best mean insertion:       {max_insertion.mean().item():.6f} m")

print()
print(f"XY aligned environments:   {int(xy_aligned.sum())} / {NUM_ENVS}")
print(f"Inserted environments:     {int(inserted.sum())} / {NUM_ENVS}")
print(f"Successful environments:   {int(success.sum())} / {NUM_ENVS}")

print()
print(f"XY alignment rate:         {xy_aligned.float().mean().item():.6f}")
print(f"Insertion rate:            {inserted.float().mean().item():.6f}")
print(f"Success rate:              {success.float().mean().item():.6f}")
print(f"Mean action magnitude:     {action_sum / NUM_STEPS:.6f}")

env.close()
simulation_app.close()

print()
print("=== M6.2.3 DIAGNOSTICS COMPLETE ===")
