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
print("M8.2 ACTION -> JOINT TARGET DIAGNOSTIC")
print("=" * 70)

print("Action dim      :", term.action_dim)
print("Command type    :", ik.cfg.command_type)
print("Relative mode   :", ik.cfg.use_relative_mode)
print("IK method       :", ik.cfg.ik_method)
print("IK params       :", ik.cfg.ik_params)
print("Scale           :", term.cfg.scale)
print("Body            :", term._body_name)
print("Joints          :", term._joint_names)

with torch.inference_mode():

    # ------------------------------------------------------------
    # 1. Get policy action
    # ------------------------------------------------------------
    actions = policy(obs)
    arm_action = actions[:, :7]

    # ------------------------------------------------------------
    # 2. Current EE frame
     # ------------------------------------------------------------
    ee_pos, ee_quat = term._compute_frame_pose()

    # ------------------------------------------------------------
    # 3. Apply exactly the same processing as ActionTerm
    # ------------------------------------------------------------
    processed_action = arm_action * term._scale

    if term.cfg.clip is not None:
        processed_action = torch.clamp(
            processed_action,
            min=term._clip[:, :, 0],
            max=term._clip[:, :, 1],
        )

    # ------------------------------------------------------------
    # 4. Set command exactly as normal action processing does
    # ------------------------------------------------------------
    ik.set_command(
        processed_action,
        ee_pos,
        ee_quat,
    )

    # ------------------------------------------------------------
    # 5. Compute Jacobian exactly as the action term does
    # ------------------------------------------------------------
    jacobian = term._compute_frame_jacobian()

    # ------------------------------------------------------------
    # 6. Current joint positions
    # ------------------------------------------------------------
    joint_pos = robot.data.joint_pos[:, term._joint_ids]

    # ------------------------------------------------------------
    # 7. Compute actual IK joint target
    # ------------------------------------------------------------
    joint_pos_des = ik.compute(
        ee_pos,
        ee_quat,
        jacobian,
        joint_pos,
    )

print()
print("=" * 70)
print("RAW PPO ACTION")
print("=" * 70)

print(arm_action[0].cpu().numpy())

print()
print("=" * 70)
print("CURRENT EE")
print("=" * 70)

print("Position:")
print(ee_pos[0].cpu().numpy())

print("Quaternion (w,x,y,z):")
print(ee_quat[0].cpu().numpy())

print()
print("=" * 70)
print("PROCESSED ACTION / TARGET POSE")
print("=" * 70)

print("Processed action:")
print(processed_action[0].cpu().numpy())

print()
print("IK target position:")
print(ik.ee_pos_des[0].cpu().numpy())

print()
print("IK target quaternion:")
print(ik.ee_quat_des[0].cpu().numpy())

print()
print("=" * 70)
print("QUATERNION CHECK")
print("=" * 70)

quat_norm = torch.linalg.vector_norm(
    ik.ee_quat_des[0]
).item()

print("Target quaternion norm:", quat_norm)

print()
print("=" * 70)
print("POSITION ERROR")
print("=" * 70)

position_error = (
    ik.ee_pos_des[0] - ee_pos[0]
)

print(position_error.cpu().numpy())

print("Position error norm:",
      torch.linalg.vector_norm(position_error).item())

print()
print("=" * 70)
print("JOINT SPACE")
print("=" * 70)

print("Current joint positions:")
print(joint_pos[0].cpu().numpy())

print()
print("IK joint target:")
print(joint_pos_des[0].cpu().numpy())

delta_joint = joint_pos_des[0] - joint_pos[0]

print()
print("Delta joint:")
print(delta_joint.cpu().numpy())

print()
print("Max |delta joint|:",
      torch.max(torch.abs(delta_joint)).item())

print()
print("=" * 70)
print("M8.2 COMPLETE")
print("=" * 70)

env.close()
simulation_app.close()
