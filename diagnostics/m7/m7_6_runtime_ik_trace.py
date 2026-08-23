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

print("====================================================")
print("M7.6 RUNTIME IK TRACE")
print("====================================================")
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
print("====================================================")

for step in range(130):

    with torch.inference_mode():
        actions = policy(obs)

        arm_action = actions[:, :7]

        ee_pos, ee_quat = term._compute_frame_pose()

        processed = arm_action * term._scale

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

    if step < 5 or 90 <= step <= 129:
        print()
        print("STEP", step + 1)
        print("PPO:", arm_action[0].cpu().numpy())
        print("PROCESSED:", processed[0].cpu().numpy())
        print("JOINT:", joint_pos[0].cpu().numpy())
        print("JOINT_DES:", joint_pos_des[0].cpu().numpy())
        print("DELTA:", (joint_pos_des[0] - joint_pos[0]).cpu().numpy())
        print("J7:", joint_pos[0, 6].item())
        print("J7_DES:", joint_pos_des[0, 6].item())
        print("J7_DELTA:", (joint_pos_des[0, 6] - joint_pos[0, 6]).item())
        print("MAX_ACTION:", arm_action.abs().max().item())
        print("MAX_PROCESSED:", processed.abs().max().item())
        print("MAX_JOINT_DES:", joint_pos_des.abs().max().item())
        print("MAX_DELTA:", (joint_pos_des - joint_pos).abs().max().item())
        print("JACOBIAN_MAX:", jacobian.abs().max().item())

    obs, rewards, dones, extras = env.step(actions)

env.close()
simulation_app.close()

print("====================================================")
print("M7.6 COMPLETE")
print("====================================================")
