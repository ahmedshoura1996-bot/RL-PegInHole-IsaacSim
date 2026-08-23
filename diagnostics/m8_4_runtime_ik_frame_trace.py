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

position_error_norms = []
quaternion_norms = []
joint_delta_maxes = []
joint_target_maxes = []
action_maxes = []
processed_maxes = []

print("=" * 70)
print("M8.4 RUNTIME IK FRAME TRACE")
print("=" * 70)
print("term:", type(term).__name__)
print("action_dim:", term.action_dim)
print("command_type:", ik.cfg.command_type)
print("relative:", ik.cfg.use_relative_mode)
print("method:", ik.cfg.ik_method)
print("params:", ik.cfg.ik_params)
print("scale:", term.cfg.scale)
print("clip:", term.cfg.clip)
print("body:", term._body_name)
print("joints:", term._joint_names)
print("=" * 70)

for step in range(130):

    with torch.inference_mode():

        actions = policy(obs)

        arm_action = actions[:, :7]

        ee_pos, ee_quat = term._compute_frame_pose()

        processed = arm_action * term._scale

        if term.cfg.clip is not None:
            processed = torch.clamp(
                processed,
                min=term._clip[:, :, 0],
                max=term._clip[:, :, 1],
            )

        ik.set_command(
            processed,
            ee_pos,
            ee_quat,
        )

        joint_pos = robot.data.joint_pos[:, term._joint_ids]

        jacobian = term._compute_frame_jacobian()

        joint_pos_des = ik.compute(
            ee_pos,
            ee_quat,
            jacobian,
            joint_pos,
        )

        position_error = ik.ee_pos_des - ee_pos

        position_error_norm = torch.linalg.vector_norm(
            position_error,
            dim=1,
        )

        quaternion_norm = torch.linalg.vector_norm(
            ik.ee_quat_des,
            dim=1,
        )

        joint_delta = joint_pos_des - joint_pos

        max_joint_delta = torch.max(
            torch.abs(joint_delta),
            dim=1,
        ).values

        max_joint_target = torch.max(
         torch.abs(joint_pos_des),
            dim=1,
        ).values

        max_action = torch.max(
            torch.abs(arm_action),
            dim=1,
        ).values

        max_processed = torch.max(
            torch.abs(processed),
            dim=1,
        ).values

    position_error_norms.append(position_error_norm[0].item())
    quaternion_norms.append(quaternion_norm[0].item())
    joint_delta_maxes.append(max_joint_delta[0].item())
    joint_target_maxes.append(max_joint_target[0].item())
    action_maxes.append(max_action[0].item())
    processed_maxes.append(max_processed[0].item())

    if step < 5 or 90 <= step <= 129:

        print()
        print("STEP", step + 1)

        print("PPO:")
        print(arm_action[0].cpu().numpy())

        print("PROCESSED:")
        print(processed[0].cpu().numpy())

        print("EE_POS:")
        print(ee_pos[0].cpu().numpy())

        print("EE_QUAT:")
        print(ee_quat[0].cpu().numpy())

        print("IK_POS_DES:")
        print(ik.ee_pos_des[0].cpu().numpy())

        print("IK_QUAT_DES:")
        print(ik.ee_quat_des[0].cpu().numpy())

        print("POSITION_ERROR_NORM:")
        print(position_error_norm[0].item())

        print("QUATERNION_NORM:")
        print(quaternion_norm[0].item())

        print("JOINT:")
        print(joint_pos[0].cpu().numpy())

        print("JOINT_DES:")
        print(joint_pos_des[0].cpu().numpy())

        print("DELTA:")
        print(joint_delta[0].cpu().numpy())

        print("MAX_ACTION:")
        print(max_action[0].item())

        print("MAX_PROCESSED:")
        print(max_processed[0].item())

        print("MAX_JOINT_DES:")
        print(max_joint_target[0].item())

        print("MAX_DELTA:")
        print(max_joint_delta[0].item())

        print("JACOBIAN_MAX:")
        print(jacobian.abs().max().item())

    obs, rewards, dones, extras = env.step(actions)


print()
print("=" * 70)
print("M8.4 RUNTIME SUMMARY")
print("=" * 70)

print()
print("POSITION ERROR NORM")
print("min :", min(position_error_norms))
print("mean:", sum(position_error_norms) / len(position_error_norms))
print("max :", max(position_error_norms))

print()
print("QUATERNION NORM")
print("min :", min(quaternion_norms))
print("mean:", sum(quaternion_norms) / len(quaternion_norms))
print("max :", max(quaternion_norms))

print()
print("MAX JOINT DELTA")
print("min :", min(joint_delta_maxes))
print("mean:", sum(joint_delta_maxes) / len(joint_delta_maxes))
print("max :", max(joint_delta_maxes))

print()
print("MAX JOINT TARGET")
print("min :", min(joint_target_maxes))
print("mean:", sum(joint_target_maxes) / len(joint_target_maxes))
print("max :", max(joint_target_maxes))

print()
print("MAX PPO ACTION")
print("min :", min(action_maxes))
print("mean:", sum(action_maxes) / len(action_maxes))
print("max :", max(action_maxes))

print()
print("MAX PROCESSED ACTION")
print("min :", min(processed_maxes))
print("mean:", sum(processed_maxes) / len(processed_maxes))
print("max :", max(processed_maxes))

print()
print("=" * 70)
print("M8.4 COMPLETE")
print("=" * 70)

env.close()
simulation_app.close()
