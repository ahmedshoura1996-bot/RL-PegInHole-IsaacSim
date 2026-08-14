from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg

import torch

print("")
print("========================================")
print("       M4.1.2 IK FRAME CHECK")
print("========================================")

cfg = PegInHoleEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 2.0

env = ManagerBasedRLEnv(cfg)

robot = env.scene["robot"]

env.reset()

# settle
zero_action = env.action_manager.action.clone()
zero_action[:] = 0.0

for _ in range(50):
    env.step(zero_action)

env.scene.update(env.sim.get_physics_dt())

# ------------------------------------------------------------
# GET IK ACTION TERM
# ------------------------------------------------------------

arm_action = env.action_manager._terms["arm_action"]

print("")
print("IK ACTION CONFIG")
print("----------------------------------------")
print("Body name:", arm_action._body_name)
print("Body index:", arm_action._body_idx)
print("Joint names:", arm_action._joint_names)
print("Joint IDs:", arm_action._joint_ids)
print("Scale:", arm_action._scale[0])
print("Offset pos:", arm_action._offset_pos[0])
print("Offset rot:", arm_action._offset_rot[0])
print("Controller:", arm_action.cfg.controller)

# ------------------------------------------------------------
# RAW PANDA HAND
# ------------------------------------------------------------

hand_pos_w = robot.data.body_pos_w[:, arm_action._body_idx].clone()
hand_quat_w = robot.data.body_quat_w[:, arm_action._body_idx].clone()

print("")
print("PANDA HAND WORLD")
print("----------------------------------------")
print("Position:", hand_pos_w[0])
print("Quaternion:", hand_quat_w[0])

# ------------------------------------------------------------
# IK FRAME
# ------------------------------------------------------------

ik_pos_b, ik_quat_b = arm_action._compute_frame_pose()

print("")
print("IK FRAME IN ROBOT ROOT FRAME")
print("----------------------------------------")
print("Position:", ik_pos_b[0])
print("Quaternion:", ik_quat_b[0])

# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------

print("")
print("ROBOT ROOT")
print("----------------------------------------")
print("Root position:", robot.data.root_pos_w[0])
print("Root quaternion:", robot.data.root_quat_w[0])

# ------------------------------------------------------------
# JACOBIAN
# ------------------------------------------------------------

jacobian = arm_action._compute_frame_jacobian()

print("")
print("JACOBIAN")
print("----------------------------------------")
print("Shape:", jacobian.shape)
print("Linear Jacobian:")
print(jacobian[0, 0:3, :])
print("Angular Jacobian:")
print(jacobian[0, 3:6, :])

# ------------------------------------------------------------
# ACTION TEST
# ------------------------------------------------------------

current_action = env.action_manager.action.clone()

print("")
print("ACTION DIMENSION")
print("----------------------------------------")
print("Action:", current_action[0])
print("Shape:", current_action.shape)

# Hold current IK frame
hold = current_action.clone()

hold[0, 0:3] = ik_pos_b[0]
hold[0, 3:7] = ik_quat_b[0]
hold[0, 7] = 0.0

print("")
print("HOLD COMMAND")
print("----------------------------------------")
print(hold[0])

# apply
for i in range(50):
    env.step(hold)
    env.scene.update(env.sim.get_physics_dt())

    if i in [0, 1, 4, 9, 24, 49]:
        hand = robot.data.body_pos_w[:, arm_action._body_idx][0]

        ik_now, _ = arm_action._compute_frame_pose()

        print(
            f"STEP {i+1:02d} | "
            f"HAND X={hand[0].item():.6f} "
            f"Y={hand[1].item():.6f} "
            f"Z={hand[2].item():.6f} | "
            f"IK X={ik_now[0,0].item():.6f} "
            f"Y={ik_now[0,1].item():.6f} "
            f"Z={ik_now[0,2].item():.6f}"
        )

print("")
print("========================================")
print("       FRAME CHECK COMPLETE")
print("========================================")

env.close()
