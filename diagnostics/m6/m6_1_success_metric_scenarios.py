from isaaclab.app import AppLauncher

app_launcher = AppLauncher({
    "headless": True,
})
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaac_lab
from isaac_lab.mdp.metrics import peg_hole_success


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

XY_TOLERANCE = 0.0005
INITIAL_PEG_Z = 0.025
INSERTION_DEPTH = 0.010


print("=== M6.1 SUCCESS METRIC SCENARIO TEST ===")
print("Environment:", ENV_NAME)

env = gym.make(ENV_NAME)
manager_env = env.unwrapped

env.reset()

peg = manager_env.scene["object"]

# Use environment-local coordinates and move the peg directly.
env_origin_z = manager_env.scene.env_origins[:, 2]


def set_peg_position(x_offset, y_offset, z):
    peg_pos = peg.data.root_pos_w.clone()

    peg_pos[:, 0] = manager_env.scene.env_origins[:, 0] + 0.5 + x_offset
    peg_pos[:, 1] = manager_env.scene.env_origins[:, 1] + y_offset
    peg_pos[:, 2] = env_origin_z + z

    peg.write_root_pose_to_sim(
 torch.cat(
            (
                peg_pos,
                peg.data.root_quat_w,
            ),
            dim=1,
        )
    )

    manager_env.sim.step()


def check_case(name, x_offset, y_offset, z, expected):
    set_peg_position(x_offset, y_offset, z)

    success = peg_hole_success(
        manager_env,
        xy_tolerance=XY_TOLERANCE,
        insertion_depth=INSERTION_DEPTH,
        initial_peg_z=INITIAL_PEG_Z,
    )

    count = int(success.sum().item())
    rate = float(success.mean().item())

    print(f"{name}:")
    print(f"  expected: {expected}")
    print(f"  success count: {count}")
    print(f"  success rate: {rate}")

    if expected == 1:
        assert count == manager_env.num_envs
    else:
        assert count == 0

    print("  PASS")


# Case 1:
# Perfect XY alignment + 10 mm insertion.
check_case(
    "CENTERED + INSERTED",
    0.0,
    0.0,
    INITIAL_PEG_Z - INSERTION_DEPTH,
    1,
)

# Case 2:
# Perfect XY alignment but insufficient insertion.
check_case(
    "CENTERED + NOT INSERTED",
    0.0,
    0.0,
    INITIAL_PEG_Z - 0.005,
    0,
)

# Case 3:
# Outside the 0.5 mm radial clearance but sufficiently inserted.
check_case(
    "OUTSIDE CLEARANCE + INSERTED",
    XY_TOLERANCE + 0.0001,
    0.0,
    INITIAL_PEG_Z - INSERTION_DEPTH,
    0,
)

# Case 4:
# Outside clearance and insufficient insertion.
check_case(
    "OUTSIDE CLEARANCE + NOT INSERTED",
    XY_TOLERANCE + 0.0001,
    0.0,
    INITIAL_PEG_Z - 0.005,
    0,
)

env.close()
simulation_app.close()

print("=== M6.1 SUCCESS METRIC SCENARIOS: PASS ===")
