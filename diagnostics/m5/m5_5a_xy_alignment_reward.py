from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab  # noqa: F401
import torch

from isaac_lab.mdp.observations import peg_hole_relative_position
from isaac_lab.mdp.rewards import peg_hole_xy_alignment


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

print("=" * 70)
print("M5.5A PEG-IN-HOLE XY ALIGNMENT REWARD VALIDATION")
print("=" * 70)

env = gym.make(
    ENV_NAME,
    num_envs=4,
)

base_env = env.unwrapped
env.reset()

peg = base_env.scene["object"]

hole_x = 0.5

cases = {
    "CENTER": (0.0, 0.0),
    "PLUS_X_5MM": (0.005, 0.0),
    "MINUS_X_5MM": (-0.005, 0.0),
    "PLUS_Y_5MM": (0.0, 0.005),
    "MINUS_Y_5MM": (0.0, -0.005),
    "PLUS_X_PLUS_Y_5MM": (0.005, 0.005),
}

results = {}

for name, (dx, dy) in cases.items():

    local_target = torch.tensor(
        [
            hole_x + dx,
            dy,
            0.025,
        ],
        device=base_env.device,
        dtype=torch.float32,
    )

    # Convert local peg position into world coordinates
    # for every replicated environment.
    target_world = (
        base_env.scene.env_origins
        + local_target.unsqueeze(0)
    )

    root_pose = torch.cat(
        [
            target_world,
            peg.data.root_quat_w.clone(),
        ],
        dim=1,
    )

    root_velocity = torch.zeros(
        (base_env.num_envs, 6),
        device=base_env.device,
        dtype=torch.float32,
    )

    peg.write_root_pose_to_sim(root_pose)
    peg.write_root_velocity_to_sim(root_velocity)

    # Update the scene state without advancing physics.
    # This is important: this diagnostic validates the mathematical
    # reward function, not the physical response of the peg.
    base_env.scene.update(base_env.cfg.sim.dt)

    relative = peg_hole_relative_position(base_env)
    reward = peg_hole_xy_alignment(base_env)

    results[name] = reward.clone()

    print(f"\\n===== {name} =====")

    print("Expected local XY offset:")
    print(
        torch.tensor(
            [dx, dy],
            device=base_env.device,
            dtype=torch.float32,
        )
    )

    print("Written world position:")
    print(peg.data.root_pos_w)

    print("Relative position:")
    print(relative)

    print("XY error:")
    print(torch.norm(relative[:, :2], dim=1))

    print("Reward:")
    print(reward)

    print("Mean:", reward.mean().item())
    print("Min :", reward.min().item())
    print("Max :", reward.max().item())


print("\n" + "=" * 70)
print("M5.5A FINAL CHECK")
print("=" * 70)

center = results["CENTER"]
plus_x = results["PLUS_X_5MM"]
minus_x = results["MINUS_X_5MM"]
plus_y = results["PLUS_Y_5MM"]
minus_y = results["MINUS_Y_5MM"]
diagonal = results["PLUS_X_PLUS_Y_5MM"]

print("Center reward:", center)
print("PLUS X 5 mm reward:", plus_x)
print("MINUS X 5 mm reward:", minus_x)
print("PLUS Y 5 mm reward:", plus_y)
print("MINUS Y 5 mm reward:", minus_y)
print("Diagonal 5/5 mm reward:", diagonal)

# --------------------------------------------------------------------------
# PASS 1: Center must give exactly 1.
# --------------------------------------------------------------------------

center_pass = bool(
    torch.allclose(
        center,
        torch.ones_like(center),
        atol=1e-6,
    )
)

# --------------------------------------------------------------------------
# PASS 2: X and Y errors of equal magnitude must be symmetric.
# --------------------------------------------------------------------------

symmetric_pass = bool(
    torch.allclose(plus_x, minus_x, atol=1e-6)
    and torch.allclose(plus_x, plus_y, atol=1e-6)
    and torch.allclose(plus_x, minus_y, atol=1e-6)
)

# --------------------------------------------------------------------------
# PASS 3: Diagonal error is sqrt(2) times larger than single-axis error,
# therefore its reward must be lower.
# --------------------------------------------------------------------------

diagonal_pass = bool(
    torch.all(diagonal < plus_x)
)

# --------------------------------------------------------------------------
# PASS 4: Center must have higher reward than any 5 mm displacement.
# --------------------------------------------------------------------------

center_max_pass = bool(
    torch.all(center > plus_x)
    and torch.all(center > plus_y)
)

print("\nPASS CONDITIONS")
print("Center reward = 1:", center_pass)
print("X/Y symmetry:", symmetric_pass)
print("Diagonal lower than single-axis:", diagonal_pass)
print("Center > 5 mm error:", center_max_pass)

all_pass = (
    center_pass
    and symmetric_pass
    and diagonal_pass
    and center_max_pass
)

print("\nM5.5A ALL PASS:", all_pass)

env.close()
simulation_app.close()
