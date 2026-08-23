from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaac_lab

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaac_lab.agents.rsl_rl_ppo_cfg import PegInHolePPORunnerCfg


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

CHECKPOINT = (
    "/workspace/projects/RL-PegInHole-IsaacSim/"
    "logs/rsl_rl/peg_in_hole/"
    "2026-08-20_13-34-34_m6_6_alignment_gated_insertion/"
    "model_60.pt"
)

env_cfg = parse_env_cfg(
    ENV_NAME,
    device="cuda:0",
    num_envs=1,
    use_fabric=True,
)

env = gym.make(ENV_NAME, cfg=env_cfg)
manager_env = env.unwrapped

env = RslRlVecEnvWrapper(env)
obs, _ = env.reset()

runner = OnPolicyRunner(
    env,
    PegInHolePPORunnerCfg().to_dict(),
    log_dir=None,
    device="cuda:0",
)

runner.load(CHECKPOINT)
policy = runner.get_inference_policy(device="cuda:0")

term = manager_env.action_manager.get_term("arm_action")
ik = term._ik_controller
robot = manager_env.scene["robot"]

print("=" * 70)
print("M8.1 RELATIVE IK DIAGNOSTIC")
print("=" * 70)

print("Action term :", type(term).__name__)
print("Action dim  :", term.action_dim)
print("Command type:", ik.cfg.command_type)
print("Relative mode:", ik.cfg.use_relative_mode)
print("IK method   :", ik.cfg.ik_method)
print("IK params   :", ik.cfg.ik_params)
print("Scale       :", term.cfg.scale)
print("Body        :", term._body_name)
print("Joints      :", term._joint_names)

print("=" * 70)
print("PPO ACTION -> CURRENT IK INTERPRETATION")
print("=" * 70)

with torch.inference_mode():
    actions = policy(obs)
    arm_action = actions[:, :7]

    ee_pos, ee_quat = term._compute_frame_pose()

print("PPO action:")
print(arm_action[0].cpu().numpy())

print()
print("Current EE position:")
print(ee_pos[0].cpu().numpy())

print()
print("Current EE quaternion (w,x,y,z):")
print(ee_quat[0].cpu().numpy())

print()
print("First 3 PPO values:")
print(arm_action[0, :3].cpu().numpy())

print()
print("Last 4 PPO values:")
print(arm_action[0, 3:].cpu().numpy())

print("=" * 70)
print("M8.1 COMPLETE")
print("=" * 70)

env.close()
simulation_app.close()
