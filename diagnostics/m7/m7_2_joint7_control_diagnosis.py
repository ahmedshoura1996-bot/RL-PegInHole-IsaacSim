from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaac_lab

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaac_lab.agents.rsl_rl_ppo_cfg import PegInHolePPORunnerCfg
from isaac_lab.mdp.observations import peg_hole_relative_position


ENV_NAME = "Isaac-PegInHole-Franka-IK-Abs-v0"

CHECKPOINT = (
    "/workspace/projects/RL-PegInHole-IsaacSim/"
    "logs/rsl_rl/peg_in_hole/"
    "2026-08-20_13-34-34_m6_6_alignment_gated_insertion/"
    "model_60.pt"
)

NUM_ENVS = 4096
NUM_STEPS = 250

XY_TOL = 0.0005
INSERTION_TARGET = 0.010


print("==============================================")
print("M7.1 ZERO-SHOT ROBUSTNESS EVALUATION")
print("==============================================")
print("Training: NONE")
print("Checkpoint: M6.6 model_60.pt")
print("Initial XY randomization: +/- 1 mm")
print("==============================================")


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

env_cfg = parse_env_cfg(
    ENV_NAME,
    device="cuda:0",
    num_envs=NUM_ENVS,
    use_fabric=True,
)

env = gym.make(ENV_NAME, cfg=env_cfg)
manager_env = env.unwrapped

print("Environment creation: PASS")

# ---------------------------------------------------------------------
# RSL-RL wrapper
# ---------------------------------------------------------------------

env = RslRlVecEnvWrapper(env)

obs, _ = env.reset()

# ---------------------------------------------------------------------
# Load M6.6 policy
# ---------------------------------------------------------------------

agent_cfg = PegInHolePPORunnerCfg()

runner = OnPolicyRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=None,
    device="cuda:0",
)

runner.load(CHECKPOINT)

policy = runner.get_inference_policy(device="cuda:0")

print("Checkpoint loaded: PASS")
print("Inference policy extracted: PASS")

print()
print("==============================================")
print("M7.2 FRANKA CONTROL CONFIGURATION")
print("==============================================")

print("Robot type:", type(env_cfg.scene.robot).__name__)

print()
print("Robot actuators:")
for name, actuator in env_cfg.scene.robot.actuators.items():
    print(f"  [{name}]")
    print("    type:", type(actuator).__name__)
    print("    joint_names:", getattr(actuator, "joint_names", None))
    print("    velocity_limit:", getattr(actuator, "velocity_limit", None))
    print("    velocity_limit_sim:", getattr(actuator, "velocity_limit_sim", None))
    print("    effort_limit:", getattr(actuator, "effort_limit", None))
    print("    effort_limit_sim:", getattr(actuator, "effort_limit_sim", None))
    print("    stiffness:", getattr(actuator, "stiffness", None))
    print("    damping:", getattr(actuator, "damping", None))

print()
print("Arm action:")
print(env_cfg.actions.arm_action)

print()

print()
print("PHYSX JOINT LIMITS")
print("----------------------------------------------")

robot = manager_env.scene["robot"]

print("Joint names:")
for i, name in enumerate(robot.joint_names):
    print(f"  {i}: {name}")

print()
print("Velocity limits [rad/s]:")
for i, value in enumerate(robot.data.joint_vel_limits[0]):
    print(f"  {robot.joint_names[i]:20s}: {value.item():.6f}")

print()
print("Effort limits [Nm]:")
for i, value in enumerate(robot.data.joint_effort_limits[0]):
    print(f"  {robot.joint_names[i]:20s}: {value.item():.6f}")

print()
print("Soft velocity limits [rad/s]:")
for i, value in enumerate(robot.data.soft_joint_vel_limits[0]):
    print(f"  {robot.joint_names[i]:20s}: {value.item():.6f}")

print("----------------------------------------------")

print("Gripper action:")
print(env_cfg.actions.gripper_action)

print("==============================================")



# ---------------------------------------------------------------------
# Initial condition diagnostics
# ---------------------------------------------------------------------

initial_rel = peg_hole_relative_position(manager_env)
initial_xy = torch.norm(initial_rel[:, :2], dim=1)

print()
print("Initial condition:")
print(
    f"  Mean XY error: {initial_xy.mean().item() * 1000:.4f} mm"
)
print(
    f"  Min XY error : {initial_xy.min().item() * 1000:.4f} mm"
)
print(
    f"  Max XY error : {initial_xy.max().item() * 1000:.4f} mm"
)
print(
    f"  Std XY error : {initial_xy.std().item() * 1000:.4f} mm"
)


# ---------------------------------------------------------------------
# Track insertion and alignment - reset-safe
# ---------------------------------------------------------------------

initial_z = manager_env.scene["object"].data.root_pos_w[:, 2].clone()

min_xy = initial_xy.clone()
max_insertion = torch.zeros(
    NUM_ENVS,
    device=manager_env.device,
)

action_sum = 0.0
action_max = 0.0
invalid_action_steps = 0

# Episode-level statistics.
episode_count = 0
episode_xy_aligned = 0
episode_inserted = 0
episode_success = 0

# Current episode state for each environment.
episode_min_xy = initial_xy.clone()
episode_max_insertion = torch.zeros(
    NUM_ENVS,
    device=manager_env.device,
)

print()
print("Running M7.1 zero-shot evaluation...")
print("Reset-safe episode tracking enabled.")

for step in range(NUM_STEPS):

    with torch.inference_mode():

        if step == 0:
            print()
            print("==============================================")
            print("M7.1 OBSERVATION / ACTION SANITY CHECK")
            print("==============================================")

            print("obs type:", type(obs))
            print("obs keys:", list(obs.keys()))

            policy_obs = obs["policy"]

            print("policy obs type:", type(policy_obs))
            print("policy obs shape:", tuple(policy_obs.shape))
            print(
                "policy obs finite:",
                torch.isfinite(policy_obs).all().item()
            )
            print(
                "policy obs min:",
                policy_obs.min().item()
            )
            print(
                "policy obs max:",
                policy_obs.max().item()
            )
            print(
                "policy obs mean:",
                policy_obs.mean().item()
            )
            print(
                "policy obs std:",
                policy_obs.std().item()
            )

        actions = policy(obs)

        # -------------------------------------------------------------
        # M7.1 ACTION EXPLOSION DETECTOR
        # -------------------------------------------------------------
        action_abs_mean = actions.abs().mean().item()
        action_abs_max = actions.abs().max().item()

        if action_abs_max > 10.0:
            policy_obs = obs["policy"]

            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("ACTION EXPLOSION DETECTED")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Step: {step + 1}")
            print(f"Action mean abs: {action_abs_mean:.6e}")
            print(f"Action max abs : {action_abs_max:.6e}")
            print(
                "Obs finite     :",
                torch.isfinite(policy_obs).all().item()
            )
            print(
                f"Obs min        : {policy_obs.min().item():.6e}"
            )
            print(
                f"Obs max        : {policy_obs.max().item():.6e}"
            )

            # ---------------------------------------------------------
            # M7.1.1 - JOINT VELOCITY FAILURE ANALYSIS
            # ---------------------------------------------------------
            # Observation layout:
            #   0:9   joint_pos
            #   9:18  joint_vel
            #   18:21 object_position
            #   21:28 target_object_position
            #   28:36 previous_actions
            #
            # Compare the actual robot joint velocities with the
            # observation values at the exact failure step.
            # ---------------------------------------------------------

            joint_pos = policy_obs[:, 0:9]
            joint_vel = policy_obs[:, 9:18]
            object_pos = policy_obs[:, 18:21]
            target_pos = policy_obs[:, 21:28]
            previous_actions = policy_obs[:, 28:36]

            print()
            print("M7.1.1 JOINT VELOCITY FAILURE ANALYSIS")
            print("----------------------------------------------")
            print(
                f"joint_pos  max abs : "
                f"{joint_pos.abs().max().item():.6e}"
            )
            print(
                f"joint_vel  max abs : "
                f"{joint_vel.abs().max().item():.6e}"
            )
            print(
                f"joint_vel  mean abs: "
                f"{joint_vel.abs().mean().item():.6e}"
            )
            print(
                f"object_pos max abs : "
     f"{object_pos.abs().max().item():.6e}"
            )
            print(
                f"target_pos max abs : "
                f"{target_pos.abs().max().item():.6e}"
            )
            print(
                f"prev_action max abs: "
                f"{previous_actions.abs().max().item():.6e}"
            )

            print()
            print("Joint velocity per DOF:")
            for j in range(9):
                print(
                    f"  joint_{j+1}: "
                    f"min={joint_vel[:, j].min().item():+.6e} "
                    f"max={joint_vel[:, j].max().item():+.6e} "
                    f"max_abs={joint_vel[:, j].abs().max().item():.6e}"
                )

            # ---------------------------------------------------------
            # Identify which observation dimensions exploded.
            # ---------------------------------------------------------
            obs_min_vals = policy_obs.min(dim=0).values
            obs_max_vals = policy_obs.max(dim=0).values

            print()
            print("Observation dimension ranges:")

            groups = [
                ("joint_pos", 0, 9),
                ("joint_vel", 9, 18),
                ("object_position", 18, 21),
         ("target_object_position", 21, 28),
                ("previous_actions", 28, 36),
            ]

            for name, start, end in groups:
                gmin = obs_min_vals[start:end].min().item()
                gmax = obs_max_vals[start:end].max().item()

                print(
                    f"{name:24s} | "
                    f"min={gmin:+.6e} | "
                    f"max={gmax:+.6e}"
                )

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            break

        if step == 0:
            print()
            print(
                "actions finite:",
                torch.isfinite(actions).all().item()
            )

            print(
                "actions min:",
                actions.min().item()
            )

            print(
                "actions max:",
                actions.max().item()
            )

            print(
                "actions mean:",
                actions.mean().item()
            )

            print(
                "actions std:",
                actions.std().item()
            )

            print("==============================================")

    # Numerical safety diagnostics.
    action_abs = actions.abs()
    action_sum += action_abs.mean().item()
    action_max = max(action_max, action_abs.max().item())

    if not torch.isfinite(actions).all():
        invalid_action_steps += 1
        print(
            f"WARNING: non-finite action detected at step {step + 1}"
        )
        break

    obs, rewards, dones, extras = env.step(actions)

    rel = peg_hole_relative_position(manager_env)
    xy_error = torch.norm(rel[:, :2], dim=1)

    current_z = manager_env.scene["object"].data.root_pos_w[:, 2]
    insertion = initial_z - current_z

    # Update current episode statistics.
    episode_min_xy = torch.minimum(
        episode_min_xy,
        xy_error,
    )

    episode_max_insertion = torch.maximum(
        episode_max_insertion,
        insertion,
    )

    # Global diagnostics.
    min_xy = torch.minimum(min_xy, xy_error)
    max_insertion = torch.maximum(max_insertion, insertion)

    # Environments whose episodes terminated/reset.
    done_ids = torch.nonzero(
        dones,
        as_tuple=False,
    ).flatten()

    if done_ids.numel() > 0:

        done_xy_aligned = (
            episode_min_xy[done_ids] <= XY_TOL
        )

        done_inserted = (
            episode_max_insertion[done_ids]
            >= INSERTION_TARGET
        )

        done_success = done_xy_aligned & done_inserted

        episode_count += done_ids.numel()
        episode_xy_aligned += done_xy_aligned.sum().item()
        episode_inserted += done_inserted.sum().item()
        episode_success += done_success.sum().item()

        # The environment has already reset these environments.
        reset_rel = peg_hole_relative_position(manager_env)
        reset_xy = torch.norm(
            reset_rel[:, :2],
            dim=1,
        )

        reset_z = manager_env.scene["object"].data.root_pos_w[:, 2]

        episode_min_xy[done_ids] = reset_xy[done_ids]
        episode_max_insertion[done_ids] = 0.0

        initial_z[done_ids] = reset_z[done_ids]

    if (step + 1) % 50 == 0:

        print(
            f"Step {step + 1:3d} | "
            f"mean XY: {xy_error.mean().item() * 1000:.4f} mm | "
            f"best XY: {min_xy.mean().item() * 1000:.4f} mm | "
            f"mean insertion: "
            f"{insertion.mean().item() * 1000:.4f} mm | "
            f"done: {done_ids.numel():4d}"
        )


# ---------------------------------------------------------------------
# Final metrics
# ---------------------------------------------------------------------

final_rel = peg_hole_relative_position(manager_env)
final_xy = torch.norm(final_rel[:, :2], dim=1)

final_z = manager_env.scene["object"].data.root_pos_w[:, 2]
final_insertion = initial_z - final_z

# Include currently active episodes.
active_xy_aligned = episode_min_xy <= XY_TOL
active_inserted = episode_max_insertion >= INSERTION_TARGET
active_success = active_xy_aligned & active_inserted

episode_count += NUM_ENVS
episode_xy_aligned += active_xy_aligned.sum().item()
episode_inserted += active_inserted.sum().item()
episode_success += active_success.sum().item()

episode_xy_alignment_rate = (
    episode_xy_aligned / episode_count
    if episode_count > 0
    else 0.0
)

episode_insertion_rate = (
    episode_inserted / episode_count
    if episode_count > 0
    else 0.0
)

episode_success_rate = (
    episode_success / episode_count
    if episode_count > 0
    else 0.0
)

mean_action = action_sum / max(NUM_STEPS, 1)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print()
print("==============================================")
print("M7.1 RESULTS")
print("==============================================")

print(
    f"Initial mean XY error:      "
    f"{initial_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Final mean XY error:        "
    f"{final_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Global best mean XY error:  "
    f"{min_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Final mean insertion:       "
    f"{final_insertion.mean().item() * 1000:.4f} mm"
)

print(
    f"Global best mean insertion: "
    f"{max_insertion.mean().item() * 1000:.4f} mm"
)

print()
print(
    f"Completed episodes:         "
    f"{episode_count}"
)

print(
    f"XY-aligned episodes:        "
    f"{episode_xy_aligned}"
)

print(
    f"Inserted episodes:          "
    f"{episode_inserted}"
)

print(
    f"Successful episodes:        "
    f"{episode_success}"
)

print()
print(
    f"Episode XY alignment rate:  "
    f"{episode_xy_alignment_rate:.6f}"
)

print(
    f"Episode insertion rate:     "
    f"{episode_insertion_rate:.6f}"
)

print(
    f"Episode success rate:       "
    f"{episode_success_rate:.6f}"
)

print(
    f"Mean action magnitude:      "
    f"{mean_action:.6f}"
)

print(
    f"Maximum action magnitude:   "
    f"{action_max:.6f}"
)

print(
    f"Invalid action steps:       "
    f"{invalid_action_steps}"
)

print("==============================================")
# ---------------------------------------------------------------------
# Final metrics
# ---------------------------------------------------------------------

final_rel = peg_hole_relative_position(manager_env)
final_xy = torch.norm(final_rel[:, :2], dim=1)

final_z = manager_env.scene["object"].data.root_pos_w[:, 2]
final_insertion = initial_z - final_z

# Include still-active episodes in the diagnostic.
active_xy_aligned = episode_min_xy <= XY_TOL
active_inserted = episode_max_insertion >= INSERTION_TARGET
active_success = active_xy_aligned & active_inserted

episode_count += NUM_ENVS
episode_xy_aligned += active_xy_aligned.sum().item()
episode_inserted += active_inserted.sum().item()
episode_success += active_success.sum().item()

episode_xy_alignment_rate = (
    episode_xy_aligned / episode_count
    if episode_count > 0
    else 0.0
)

episode_insertion_rate = (
    episode_inserted / episode_count
    if episode_count > 0
    else 0.0
)

episode_success_rate = (
    episode_success / episode_count
    if episode_count > 0
    else 0.0
)

mean_action = action_sum / max(NUM_STEPS, 1)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print()
print("==============================================")
print("M7.1 RESULTS")
print("==============================================")

print(
    f"Initial mean XY error:     "
    f"{initial_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Final mean XY error:       "
    f"{final_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Global best mean XY error: "
    f"{min_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Final mean insertion:      "
    f"{final_insertion.mean().item() * 1000:.4f} mm"
)

print(
    f"Global best mean insertion:"
    f" {max_insertion.mean().item() * 1000:.4f} mm"
)

print()
print(
    f"Completed episodes:        "
    f"{episode_count}"
)

print(
    f"XY-aligned episodes:       "
    f"{episode_xy_aligned}"
)

print(
    f"Inserted episodes:         "
    f"{episode_inserted}"
)

print(
    f"Successful episodes:       "
    f"{episode_success}"
)

print()
print(
    f"Episode XY alignment rate: "
    f"{episode_xy_alignment_rate:.6f}"
)

print(
    f"Episode insertion rate:     "
    f"{episode_insertion_rate:.6f}"
)

print(
    f"Mean action magnitude:      "
    f"{mean_action:.6f}"
)

print(
    f"Maximum action magnitude:   "
    f"{action_max:.6f}"
)

print(
    f"Invalid action steps:       "
    f"{invalid_action_steps}"
)

print("==============================================")
# ---------------------------------------------------------------------
# Final metrics
# ---------------------------------------------------------------------

final_rel = peg_hole_relative_position(manager_env)
final_xy = torch.norm(final_rel[:, :2], dim=1)

final_z = manager_env.scene["object"].data.root_pos_w[:, 2]
final_insertion = initial_z - final_z


xy_aligned = min_xy <= XY_TOL
inserted = max_insertion >= INSERTION_TARGET

success = xy_aligned & inserted


xy_alignment_rate = xy_aligned.float().mean()
insertion_rate = inserted.float().mean()
success_rate = success.float().mean()

mean_action = action_sum / NUM_STEPS


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print()
print("==============================================")
print("M7.1 RESULTS")
print("==============================================")

print(
    f"Initial mean XY error:     "
    f"{initial_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Final mean XY error:       "
    f"{final_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Best mean XY error:        "
    f"{min_xy.mean().item() * 1000:.4f} mm"
)

print(
    f"Final mean insertion:      "
    f"{final_insertion.mean().item() * 1000:.4f} mm"
)

print(
    f"Best mean insertion:       "
    f"{max_insertion.mean().item() * 1000:.4f} mm"
)

print()

print(
    f"XY aligned environments:   "
    f"{xy_aligned.sum().item()} / {NUM_ENVS}"
)

print(
    f"Inserted environments:     "
    f"{inserted.sum().item()} / {NUM_ENVS}"
)

print(
    f"Successful environments:   "
    f"{success.sum().item()} / {NUM_ENVS}"
)

print()

print(
    f"XY alignment rate:         "
    f"{xy_alignment_rate.item():.6f}"
)

print(
    f"Insertion rate:            "
    f"{insertion_rate.item():.6f}"
)

print(
    f"Success rate:              "
"Success rate:              "
    f"{success_rate.item():.6f}"
)

print(
    f"Mean action magnitude:     "
    f"{mean_action:.6f}"
)

print("==============================================")

env.close()
simulation_app.close()
