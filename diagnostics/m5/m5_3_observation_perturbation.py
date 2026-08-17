from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab  # noqa: F401
import torch

from isaac_lab.mdp.observations import peg_hole_relative_position


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"


print("=" * 70)
print("M5.3 OBSERVATION PERTURBATION VALIDATION")
print("=" * 70)

env = gym.make(
    ENV_NAME,
    num_envs=4,
)

base_env = env.unwrapped
obs, info = env.reset()

peg = base_env.scene["object"]


def check_case(name, delta_xy, expected_xy):
    """Apply controlled XY perturbation and validate the observation."""

    delta = torch.tensor(
        [delta_xy[0], delta_xy[1], 0.0],
        device=base_env.device,
        dtype=peg.data.root_pos_w.dtype,
    )

    # Apply perturbation in world coordinates.
    peg.data.root_pos_w[:, :3] = (
        peg.data.root_pos_w[:, :3] + delta
      )

    # Read custom observation.
    relative = peg_hole_relative_position(base_env)

    expected = torch.tensor(
        [expected_xy[0], expected_xy[1], 0.025],
        device=base_env.device,
        dtype=relative.dtype,
    )

    expected = expected.unsqueeze(0).repeat(
        base_env.num_envs, 1
    )

    error = torch.abs(relative - expected)
    max_error = error.max().item()

    xy_error = torch.norm(
        relative[:, :2],
        dim=1,
    )

    passed = torch.all(
        error < 1e-4
    ).item()

    print(f"\n===== {name} =====")
    print("Expected:")
    print(expected)

    print("Measured:")
    print(relative)

    print("XY magnitude:")
    print(xy_error)

    print("Max absolute error:", max_error)

    print("PASS:", bool(passed))

    return bool(passed)


results = []


test_cases = [
    ("CENTER", (0.000, 0.000), (0.000, 0.000)),
    ("PLUS_X_5MM", (0.005, 0.000), (0.005, 0.000)),
    ("MINUS_X_5MM", (-0.005, 0.000), (-0.005, 0.000)),
     ("PLUS_Y_5MM", (0.000, 0.005), (0.000, 0.005)),
    ("MINUS_Y_5MM", (0.000, -0.005), (0.000, -0.005)),
    ("PLUS_X_PLUS_Y_5MM", (0.005, 0.005), (0.005, 0.005)),
]


for name, delta, expected in test_cases:

    # Restore nominal peg position before each independent test.
    env.reset()

    results.append(
        check_case(
            name,
            delta,
            expected,
        )
    )


print("\n" + "=" * 70)
print("M5.3 FINAL RESULT")
print("=" * 70)

print(
    "Cases passed:",
    sum(results),
    "/",
    len(results),
)

print(
    "ALL CASES PASS:",
    all(results),
)

assert all(results), "M5.3 perturbation validation FAILED"


env.close()
simulation_app.close()
