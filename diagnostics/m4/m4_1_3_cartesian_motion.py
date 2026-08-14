from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg

import torch

print("")
print("========================================")
print("       M4.1.3 CARTESIAN Z MOTION")
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

# ------------------------------------------------------------
# +Z 10 mm CARTESIAN MOTION TEST
# ------------------------------------------------------------

target_z = ik_pos_b[0, 2] + 0.010

action_z = current_action.clone()

action_z[0, 0] = ik_pos_b[0, 0]
action_z[0, 1] = ik_pos_b[0, 1]
action_z[0, 2] = target_z
action_z[0, 3:7] = ik_quat_b[0]
action_z[0, 7] = 0.0

print("")
print("+Z 10 MM COMMAND")
print("----------------------------------------")
print(f"Initial IK Z = {ik_pos_b[0,2].item():.6f} m")
print(f"Target IK Z  = {target_z.item():.6f} m")
print(f"Requested ΔZ = {target_z.item() - ik_pos_b[0,2].item():.6f} m")
print(action_z[0])

# ------------------------------------------------------------
# APPLY CARTESIAN TARGET
# ------------------------------------------------------------

for i in range(100):
    env.step(action_z)
    env.scene.update(env.sim.get_physics_dt())

    if i in [0, 1, 4, 9, 24, 49, 74, 99]:
        hand = robot.data.body_pos_w[:, arm_action._body_idx][0]
        ik_now, _ = arm_action._compute_frame_pose()

        print(
            f"STEP {i+1:03d} | "
            f"HAND X={hand[0].item():.6f} "
            f"Y={hand[1].item():.6f} "
            f"Z={hand[2].item():.6f} | "
            f"IK X={ik_now[0,0].item():.6f} "
            f"Y={ik_now[0,1].item():.6f} "
            f"Z={ik_now[0,2].item():.6f} | "
            f"ΔZ={(ik_now[0,2].item() - ik_pos_b[0,2].item()):.6f}"
        )

# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

ik_final, _ = arm_action._compute_frame_pose()

final_dz = ik_final[0, 2] - ik_pos_b[0, 2]
final_error = final_dz - 0.010

print("")
print("FINAL RESULT")
print("----------------------------------------")
print(f"Initial IK Z : {ik_pos_b[0,2].item():.6f} m")
print(f"Target IK Z  : {target_z.item():.6f} m")
print(f"Final IK Z   : {ik_final[0,2].item():.6f} m")
print(f"Actual ΔZ    : {final_dz.item():.6f} m")
print(f"Error        : {final_error.item():.6f} m")

if abs(final_error.item()) <= 0.001:
    print("RESULT: PASS — Cartesian +Z 10 mm motion succeeded.")
else:
    print("RESULT: CHECK — Cartesian motion did not reach within ±1 mm.")

print("")
print("========================================")
print("       FRAME CHECK COMPLETE")
print("========================================")

env.close()
