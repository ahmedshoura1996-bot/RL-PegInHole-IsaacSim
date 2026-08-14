from isaaclab.app import AppLauncher

# ============================================================
# M4.2 - FIRST VISUAL SIMULATION
# ============================================================

app_launcher = AppLauncher(livestream=1)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg

import torch


print("")
print("========================================")
print("       M4.2 FIRST VISUAL SIMULATION")
print("========================================")


# ============================================================
# ENVIRONMENT
# ============================================================

cfg = PegInHoleEnvCfg()

cfg.scene.num_envs = 1
cfg.scene.env_spacing = 2.0

env = ManagerBasedRLEnv(cfg)

robot = env.scene["robot"]


# ============================================================
# RESET
# ============================================================

env.reset()

print("")
print("Environment reset successfully.")


# ============================================================
# SETTLE ROBOT
# ============================================================

zero_action = env.action_manager.action.clone()
zero_action[:] = 0.0

for _ in range(50):
    env.step(zero_action)

env.scene.update(env.sim.get_physics_dt())


# ============================================================
# GET IK ACTION TERM
# ============================================================

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


# ============================================================
# CURRENT IK FRAME
# ============================================================

ik_pos_b, ik_quat_b = arm_action._compute_frame_pose()

print("")
print("INITIAL IK FRAME")
print("----------------------------------------")
print("Position:", ik_pos_b[0])
print("Quaternion:", ik_quat_b[0])


# ============================================================
# CREATE CARTESIAN TARGET
# ============================================================

target_pos = ik_pos_b.clone()
target_pos[0, 2] += 0.010

target_quat = ik_quat_b.clone()


print("")
print("CARTESIAN TARGET")
print("----------------------------------------")
print("Initial X:", ik_pos_b[0, 0].item())
print("Initial Y:", ik_pos_b[0, 1].item())
print("Initial Z:", ik_pos_b[0, 2].item())

print("")
print("Target X:", target_pos[0, 0].item())
print("Target Y:", target_pos[0, 1].item())
print("Target Z:", target_pos[0, 2].item())

print("")
print("Requested displacement: +10 mm")


# ============================================================
# BUILD CARTESIAN ACTION
# ============================================================

current_action = env.action_manager.action.clone()

target_action = current_action.clone()

target_action[0, 0] = target_pos[0, 0]
target_action[0, 1] = target_pos[0, 1]
target_action[0, 2] = target_pos[0, 2]

target_action[0, 3:7] = target_quat[0]

# Gripper command remains zero.
target_action[0, 7] = 0.0


print("")
print("TARGET ACTION")
print("----------------------------------------")
print(target_action[0])


# ============================================================
# VISUAL MOTION
# ============================================================

print("")
print("========================================")
print("       STARTING VISUAL MOTION")
print("========================================")

for i in range(300):

    env.step(target_action)

    env.scene.update(env.sim.get_physics_dt())

    if i in [0, 9, 24, 49, 99, 149, 199, 249, 299]:

        ik_now, _ = arm_action._compute_frame_pose()

        dz = (
            ik_now[0, 2].item()
            - ik_pos_b[0, 2].item()
        )

        error = (
            target_pos[0, 2].item()
            - ik_now[0, 2].item()
        )

        print(
            f"STEP {i+1:03d} | "
            f"IK X={ik_now[0,0].item():.6f} | "
            f"IK Y={ik_now[0,1].item():.6f} | "
            f"IK Z={ik_now[0,2].item():.6f} | "
            f"DeltaZ={dz*1000:.3f} mm | "
            f"Z error={error*1000:.3f} mm"
        )


# ============================================================
# FINAL MEASUREMENT
# ============================================================

ik_final, _ = arm_action._compute_frame_pose()

final_dz = (
    ik_final[0, 2].item()
    - ik_pos_b[0, 2].item()
)

final_error = (
    target_pos[0, 2].item()
    - ik_final[0, 2].item()
)


print("")
print("========================================")
print("       M4.2 FINAL RESULT")
print("========================================")

print(f"Initial IK Z : {ik_pos_b[0,2].item():.6f} m")
print(f"Target IK Z  : {target_pos[0,2].item():.6f} m")
print(f"Final IK Z   : {ik_final[0,2].item():.6f} m")

print("")
print(f"Requested DeltaZ : 10.000 mm")
print(f"Actual DeltaZ    : {final_dz * 1000:.3f} mm")
print(f"Final error      : {final_error * 1000:.3f} mm")


# ============================================================
# PASS / CHECK
# ============================================================

if abs(final_error) <= 0.001:

    print("")
    print("RESULT: PASS")
    print("Visual Cartesian motion reached the target within 1 mm.")

else:

    print("")
    print("RESULT: CHECK")
    print("Cartesian motion did not reach the target within 1 mm.")


print("")
print("========================================")
print("       M4.2 VISUAL TEST COMPLETE")
print("========================================")


# ============================================================
# KEEP ISAAC SIM OPEN
# ============================================================

print("")
print("Isaac Sim remains open for visual inspection.")
print("Close the simulator manually when finished.")


while simulation_app.is_running():

    simulation_app.update()


env.close()
simulation_app.close()
