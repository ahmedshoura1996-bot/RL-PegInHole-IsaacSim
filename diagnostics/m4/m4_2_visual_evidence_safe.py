from isaaclab.app import AppLauncher

# ============================================================
# M4.2 - SAFE VISUAL EVIDENCE CAPTURE
# Captures ONLY 3 PNG images:
#   1) initial
#   2) middle
#   3) final
# ============================================================

app_launcher = AppLauncher(livestream=1)
simulation_app = app_launcher.app

import os
import time
import numpy as np
from PIL import Image
import omni.replicator.core as rep

from isaaclab.envs import ManagerBasedRLEnv
from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg


OUTPUT_DIR = "/workspace/projects/RL-PegInHole-IsaacSim/artifacts/m4/m4_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("")
print("========================================")
print("       M4.2 SAFE VISUAL EVIDENCE")
print("========================================")
print("Output:", OUTPUT_DIR)


# ============================================================
# ENVIRONMENT
# ============================================================

cfg = PegInHoleEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 2.0

env = ManagerBasedRLEnv(cfg)
env.reset()

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
# IK TARGET
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
# RGB ANNOTATOR
# NO WRITER
# NO CONTINUOUS RECORDING
# ============================================================

rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])


def capture_frame(filename):

    print("")
    print("Capturing:", filename)

    # Advance rendering once.
    simulation_app.update()

    # Explicitly request ONE Replicator capture.
    rep.orchestrator.step(
        rt_subframes=1,
        pause_timeline=True
    )

    # Allow render result to become available.
    simulation_app.update()

    data = rgb_annotator.get_data()

    if data is None:
        raise RuntimeError("RGB annotator returned no image data.")

    # Remove alpha if present.
    if data.shape[-1] == 4:
        data = data[:, :, :3]

    image = Image.fromarray(
        np.asarray(data, dtype=np.uint8)
    )

    path = os.path.join(OUTPUT_DIR, filename)

    image.save(path)

    print("Saved:", path)
    print("Size:", image.size)

    return path


# ============================================================
# INITIAL IMAGE
# ============================================================

capture_frame("M4_2_initial.png")


# ============================================================
# VISUAL MOTION
# ============================================================

print("")
print("========================================")
print("       STARTING +10 mm MOTION")
print("========================================")

capture_steps = 100
middle_step = 50

for i in range(capture_steps):

    env.step(target_action)
    env.scene.update(env.sim.get_physics_dt())

    simulation_app.update()

    if i + 1 == middle_step:

        ik_middle, _ = arm_action._compute_frame_pose()

        middle_dz = (
            ik_middle[0, 2].item()
            - ik_pos_b[0, 2].item()
        )

        print(
            f"MIDDLE STEP {i+1:03d} | "
            f"DeltaZ={middle_dz * 1000:.3f} mm"
        )

        capture_frame("M4_2_middle.png")


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


# ============================================================
# FINAL IMAGE
# ============================================================

capture_frame("M4_2_final.png")


# ============================================================
# RESULTS FILE
# ============================================================

result_path = os.path.join(
    OUTPUT_DIR,
    "M4_2_results.txt"
)

with open(result_path, "w") as f:

    f.write("M4.2 VISUAL CARTESIAN MOTION EVIDENCE\n")
    f.write("========================================\n\n")

    f.write(f"Initial Z : {ik_pos_b[0,2].item():.9f} m\n")
    f.write(f"Target Z  : {target_pos[0,2].item():.9f} m\n")
    f.write(f"Final Z   : {ik_final[0,2].item():.9f} m\n\n")

    f.write("Requested DeltaZ : 10.000 mm\n")
    f.write(f"Actual DeltaZ    : {final_dz * 1000:.3f} mm\n")
    f.write(f"Final error      : {final_error * 1000:.3f} mm\n\n")

    if abs(final_error) <= 0.001:
        f.write("RESULT: PASS\n")
        f.write("Cartesian motion reached the target within 1 mm.\n")
    else:
        f.write("RESULT: CHECK\n")
        f.write("Cartesian motion did not reach the target within 1 mm.\n")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("")
print("========================================")
print("       M4.2 EVIDENCE COMPLETE")
print("========================================")

print(f"Initial Z : {ik_pos_b[0,2].item():.6f} m")
print(f"Target Z  : {target_pos[0,2].item():.6f} m")
print(f"Final Z   : {ik_final[0,2].item():.6f} m")

print("")
print(f"Requested DeltaZ : 10.000 mm")
print(f"Actual DeltaZ    : {final_dz * 1000:.3f} mm")
print(f"Final error      : {final_error * 1000:.3f} mm")

if abs(final_error) <= 0.001:
    print("RESULT: PASS")
else:
    print("RESULT: CHECK")

print("")
print("Files:")
print("  M4_2_initial.png")
print("  M4_2_middle.png")
print("  M4_2_final.png")
print("  M4_2_results.txt")

print("")
print("Cleaning up...")

rgb_annotator.detach()
env.close()
simulation_app.close()

print("DONE")
