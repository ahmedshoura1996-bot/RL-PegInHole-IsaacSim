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

all_actions = []
all_quat_norms = []
all_position_errors = []

print("=" * 70)
print("M8.3 ACTION DISTRIBUTION DIAGNOSTIC")
print("=" * 70)

print("Command type  :", ik.cfg.command_type)
print("Relative mode :", ik.cfg.use_relative_mode)
print("IK method     :", ik.cfg.ik_method)
print("Action dim    :", term.action_dim)

with torch.inference_mode():

    for i in range(20):

        actions = policy(obs)
        arm_action = actions[:, :7]

        ee_pos, ee_quat = term._compute_frame_pose()

        processed_action = arm_action * term._scale

        ik.set_command(
            processed_action,
            ee_pos,
            ee_quat,
        )

        position_error = ik.ee_pos_des - ee_pos

        quat_norm = torch.linalg.vector_norm(
            ik.ee_quat_des,
            dim=1,
        )

        all_actions.append(
            arm_action[0].detach().cpu()
        )

        all_quat_norms.append(
            quat_norm[0].item()
        )

        all_position_errors.append(
            torch.linalg.vector_norm(
                position_error,
                dim=1
            )[0].item()
        )

        obs, _, _, _ = env.step(
            torch.cat(
                (
                    arm_action,
                    torch.zeros((1, 1), device=arm_action.device),
                ),
                dim=1,
            )
        )

        if (i + 1) % 5 == 0:
            print(f"Collected {i + 1}/20 samples")

actions_tensor = torch.stack(all_actions)

print()
print("=" * 70)
print("ARM ACTION STATISTICS")
print("=" * 70)

for j in range(7):
    print(
        f"Action {j}: "
        f"min={actions_tensor[:, j].min().item(): .5f}, "
        f"max={actions_tensor[:, j].max().item(): .5f}, "
        f"mean={actions_tensor[:, j].mean().item(): .5f}, "
        f"std={actions_tensor[:, j].std().item(): .5f}"
    )

print()
print("=" * 70)
print("QUATERNION NORM")
print("=" * 70)

print(
    "min :",
    min(all_quat_norms)
)

print(
    "max :",
    max(all_quat_norms)
)

print(
    "mean:",
    sum(all_quat_norms) / len(all_quat_norms)
)

print()
print("=" * 70)
print("POSITION ERROR NORM")
print("=" * 70)

print(
    "min :",
    min(all_position_errors)
)

print(
    "max :",
    max(all_position_errors)
)

print(
    "mean:",
    sum(all_position_errors) / len(all_position_errors)
)

print()
print("=" * 70)
print("M8.3 COMPLETE")
print("=" * 70)

env.close()
simulation_app.close()
