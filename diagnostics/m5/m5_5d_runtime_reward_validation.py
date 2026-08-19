from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

EXPECTED_WEIGHTS = {
    "peg_hole_xy_alignment": 0.5,
    "peg_insertion_progress": 0.5,
    "action_rate": -1e-4,
    "joint_vel": -1e-4,
}

DISABLED_LIFT_REWARDS = {
    "reaching_object",
    "lifting_object",
    "object_goal_tracking",
    "object_goal_tracking_fine_grained",
}

print("=" * 70)
print("M5.7 RUNTIME REWARD CONFIGURATION VALIDATION")
print("=" * 70)

env = gym.make(
    ENV_NAME,
    num_envs=4,
)

base_env = env.unwrapped

term_names = list(base_env.reward_manager._term_names)
term_cfgs = list(base_env.reward_manager._term_cfgs)

actual_weights = {
    name: cfg.weight
    for name, cfg in zip(term_names, term_cfgs)
}

print("\nActive reward terms:")
for name in term_names:
    print(f"  {name}")

print("\nActive reward weights:")
for name, weight in actual_weights.items():
    print(f"  {name} = {weight}")

# ----------------------------------------------------------------------
# PASS 1: Required Peg-in-Hole rewards are active.
# ----------------------------------------------------------------------

required_terms_pass = all(
    name in term_names
    for name in EXPECTED_WEIGHTS
)

# ----------------------------------------------------------------------
# PASS 2: Generic Lift rewards are disabled.
# ----------------------------------------------------------------------

disabled_terms_pass = all(
    name not in term_names
    for name in DISABLED_LIFT_REWARDS
)

# ----------------------------------------------------------------------
# PASS 3: Reward weights match M5.6 configuration.
# ----------------------------------------------------------------------

weights_pass = all(
    name in actual_weights
    and abs(actual_weights[name] - expected_weight) < 1e-8
    for name, expected_weight in EXPECTED_WEIGHTS.items()
)

# ----------------------------------------------------------------------
# Final result.
# ----------------------------------------------------------------------

all_pass = (
    required_terms_pass
    and disabled_terms_pass
    and weights_pass
)

print("\n" + "=" * 70)
print("M5.7 PASS CONDITIONS")
print("=" * 70)

print("Required Peg-in-Hole rewards active:", required_terms_pass)
print("Generic Lift rewards disabled:", disabled_terms_pass)
print("Reward weights correct:", weights_pass)

print("\nM5.7 ALL PASS:", all_pass)

env.close()
simulation_app.close()
