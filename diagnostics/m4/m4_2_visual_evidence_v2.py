from isaaclab.app import AppLauncher

# ============================================================
# M4.2 - VISUAL EVIDENCE V2
# Direct Isaac Sim Camera API
# NO Replicator Writer
# NO continuous image capture
# ============================================================

app_launcher = AppLauncher(livestream=1)
simulation_app = app_launcher.app

import os
import numpy as np
from PIL import Image

from isaacsim.sensors.camera import Camera
from isaaclab.envs import ManagerBasedRLEnv
from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg


print("")
print("========================================")
print("       M4.2 VISUAL EVIDENCE V2")
print("========================================")
print("Direct Camera API - 3 images only")
print("========================================")


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "/workspace/projects/RL-PegInHole-IsaacSim/artifacts/m4/m4_2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clean previous evidence only
for f in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(path):
        os.remove(path)

print("Output:", OUTPUT_DIR)


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

print("Environment settled.")


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

print("")
print("Creating Isaac Sim camera...")

camera = Camera(
    prim_path="/World/M4_2_EvidenceCamera",
    position=np.array([1.0, -1.0, 0.85]),
    resolution=(1280, 720),
    frequency=50,
)

camera.set_focal_length(24.0)

camera.initialize()

# Point camera toward workspace
camera.set_world_pose(
    position=np.array([1.0, -1.0, 0.85]),
    orientation=None,
)

print("Camera initialized.")


# ============================================================
# CAMERA WARM-UP
# ============================================================

print("")
print("Warming up renderer...")

for _ in range(10):
    simulation_app.update()

print("Renderer ready.")


# ============================================================
# IMAGE SAVE FUNCTION
# ============================================================

def capture_image(filename):

    print("")
    print("Capturing:", filename)

    image = camera.get_rgba()

    if image is None:
        raise RuntimeError("Camera returned None")

    image = np.asarray(image)

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    if image.size == 0:
        raise RuntimeError("Camera returned empty image")

    # Remove alpha channel
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.ascontiguousarray(image.astype(np.uint8))

    output_path = os.path.join(OUTPUT_DIR, filename)

    Image.fromarray(image).save(output_path)

    print("Saved:", output_path)


# ============================================================
# INITIAL FRAME
# ============================================================

capture_image("M4_2_initial.png")


# ============================================================
# MOTION
# ============================================================

print("")
print("========================================")
print("       STARTING 10 MM MOTION")
print("========================================")

capture_steps = 300

checkpoints = [0, 49, 99, 149, 199, 249, 299]

for i in range(capture_steps):

    env.step(target_action)

    if i in checkpoints:

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
            f"DeltaZ={dz * 1000:.3f} mm | "
            f"Error={error * 1000:.3f} mm"
        )

    simulation_app.update()


# ============================================================
# FINAL FRAME
# ============================================================

capture_image("M4_2_final.png")


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
# RESULT
# ============================================================

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
# FILE CHECK
# ============================================================

print("")
print("========================================")
print("       EVIDENCE FILES")
print("========================================")

for f in sorted(os.listdir(OUTPUT_DIR)):

    path = os.path.join(OUTPUT_DIR, f)

    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)

        print(
            f"{f} | "
            f"{size_mb:.2f} MB"
        )


# ============================================================
# CLEANUP
# ============================================================

print("")
print("Closing environment...")

env.close()

simulation_app.close()

print("")
print("========================================")
print("       M4.2 COMPLETE")
print("========================================")
