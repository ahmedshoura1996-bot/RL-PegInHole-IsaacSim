from isaaclab.app import AppLauncher

# ============================================================
# M4.2 - VISUAL EVIDENCE CAPTURE
# ============================================================

app_launcher = AppLauncher(livestream=1)
simulation_app = app_launcher.app

import os
import time
import torch
import omni.replicator.core as rep

from isaaclab.envs import ManagerBasedRLEnv
from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg


print("")
print("========================================")
print("       M4.2 VISUAL EVIDENCE CAPTURE")
print("========================================")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "/workspace/projects/RL-PegInHole-IsaacSim/artifacts/m4/m4_2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Output directory:", OUTPUT_DIR)


# ============================================================
# ENVIRONMENT
# ============================================================

cfg = PegInHoleEnvCfg()

cfg.scene.num_envs = 1
cfg.scene.env_spacing = 2.0

env = ManagerBasedRLEnv(cfg)

robot = env.scene["robot"]

env.reset()

print("")
print("Environment reset successfully.")


# ============================================================
# SETTLE
# ============================================================

zero_action = env.action_manager.action.clone()
zero_action[:] = 0.0

for _ in range(50):
    env.step(zero_action)

env.scene.update(env.sim.get_physics_dt())


# ============================================================
# IK ACTION
# ============================================================

arm_action = env.action_manager._terms["arm_action"]

ik_pos_b, ik_quat_b = arm_action._compute_frame_pose()

target_pos = ik_pos_b.clone()
target_pos[0, 2] += 0.010

target_quat = ik_quat_b.clone()

current_action = env.action_manager.action.clone()
target_action = current_action.clone()

target_action[0, 0] = target_pos[0, 0]
target_action[0, 1] = target_pos[0, 1]
target_action[0, 2] = target_pos[0, 2]
target_action[0, 3:7] = target_quat[0]
target_action[0, 7] = 0.0


# ============================================================
# CAMERA
# ============================================================

print("")
print("Creating evidence camera...")

camera = rep.create.camera(
    position=(1.0, -1.0, 0.85),
    look_at=(0.20, 0.05, 0.35)
)

render_product = rep.create.render_product(
    camera,
    (1280, 720)
)


# ============================================================
# IMAGE WRITER
# ============================================================

writer = rep.WriterRegistry.get("BasicWriter")

writer.initialize(
    output_dir=OUTPUT_DIR,
    rgb=True
)

writer.attach([render_product])


# ============================================================
# CAPTURE INITIAL FRAME
# ============================================================

print("")
print("Capturing INITIAL frame...")

for _ in range(5):
    simulation_app.update()

rep.orchestrator.step()

time.sleep(1)


# ============================================================
# VISUAL MOTION
# ============================================================

print("")
print("========================================")
print("       STARTING CAPTURED MOTION")
print("========================================")

capture_steps = 300

for i in range(capture_steps):

    env.step(target_action)
    env.scene.update(env.sim.get_physics_dt())

    simulation_app.update()

    # Capture every 2 simulation steps
    if i % 2 == 0:
        rep.orchestrator.step()

    if i in [0, 49, 99, 149, 199, 249, 299]:
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
            f"DeltaZ={dz*1000:.3f} mm | "
            f"Error={error*1000:.3f} mm"
        )


# ============================================================
# FINAL FRAME
# ============================================================

print("")
print("Capturing FINAL frame...")

for _ in range(5):
    simulation_app.update()

rep.orchestrator.step()

time.sleep(2)


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
print("       M4.2 CAPTURE RESULT")
print("========================================")

print(f"Initial Z : {ik_pos_b[0,2].item():.6f} m")
print(f"Target Z  : {target_pos[0,2].item():.6f} m")
print(f"Final Z   : {ik_final[0,2].item():.6f} m")

print("")
print(f"Requested DeltaZ : 10.000 mm")
print(f"Actual DeltaZ    : {final_dz * 1000:.3f} mm")
print(f"Final error      : {final_error * 1000:.3f} mm")

if abs(final_error) <= 0.001:
    print("")
    print("RESULT: PASS")
else:
    print("")
    print("RESULT: CHECK")


# ============================================================
# CLEANUP
# ============================================================

print("")
print("Flushing captured frames...")

try:
    rep.orchestrator.wait_until_complete()
except Exception as e:
    print("Replicator flush warning:", e)

time.sleep(2)

writer.detach()

env.close()

print("")
print("========================================")
print("       CAPTURE COMPLETE")
print("========================================")

print("Evidence directory:")
print(OUTPUT_DIR)

print("")
print("Isaac Sim remains open.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
