import argparse
import os
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="M9.3 DDPG checkpoint evaluation"
)

parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--num_steps", type=int, default=250)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", required=True)

parser.add_argument(
    "--output",
    default=(
        "results/benchmark_v1/ddpg/"
        "benchmark_10k_safe_noise/m9_eval.txt"
    ),
)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()


# ============================================================
# DDPG policy
# IMPORTANT: Must exactly match the training architecture.
# 36 -> 256 -> 256 -> 256 -> 8
# ============================================================

class TD3Policy(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ELU(),

            torch.nn.Linear(256, 256),
            torch.nn.ELU(),

            torch.nn.Linear(256, 256),
            torch.nn.ELU(),

            torch.nn.Linear(256, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, observations):
        return self.net(observations)

    def deterministic_action(self, observations):
        return self.forward(observations)


# ============================================================
# CRITICAL:
# Load policy BEFORE Isaac Sim starts.
# ============================================================

print("=" * 70, flush=True)
print("TD3 M9.5 PRELOAD", flush=True)
print("=" * 70, flush=True)
print(f"Checkpoint : {args.checkpoint}", flush=True)

print("Loading policy BEFORE Isaac Sim...", flush=True)

if not os.path.isfile(args.checkpoint):
    raise FileNotFoundError(
        f"Checkpoint not found: {args.checkpoint}"
    )

checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=True,
)

if "policy" not in checkpoint:
    raise RuntimeError(
        "TD3 checkpoint does not contain a 'policy' state_dict"
    )

policy_state = checkpoint["policy"]

preloaded_policy = TD3Policy(
    obs_dim=36,
    action_dim=8,
)

preloaded_policy.load_state_dict(
    policy_state,
    strict=True,
)

preloaded_policy.eval()

for name, tensor in preloaded_policy.state_dict().items():
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Non-finite policy tensor: {name}"
        )

print("Policy preload: PASS", flush=True)


# ============================================================
# NOW start Isaac Sim
# ============================================================

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ============================================================
# IsaacLab imports AFTER AppLauncher
# ============================================================

import gymnasium as gym

import isaac_lab
import isaaclab_tasks

from isaaclab_tasks.utils import parse_env_cfg
from isaac_lab.mdp.observations import peg_hole_relative_position


ENV_ID = "Isaac-PegInHole-Franka-IK-Abs-v0"
XY_TOL = 0.0005
INSERTION_TARGET = 0.010


def get_policy_obs(obs):

    if isinstance(obs, dict):

        if "policy" in obs:
            return obs["policy"]

        if "observations" in obs:
            return obs["observations"]

    raise RuntimeError(
        f"Unexpected observation structure: {type(obs)}"
    )



def main():

    device = "cuda:0"

    print("=" * 70)
    print("TD3 M9.5 CHECKPOINT EVALUATION")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Num envs   : {args.num_envs}")
    print(f"Num steps  : {args.num_steps}")
    print(f"Seed       : {args.seed}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env_cfg = parse_env_cfg(
        ENV_ID,
        device=device,
        num_envs=args.num_envs,
        use_fabric=True,
    )

    env = gym.make(
        ENV_ID,
        cfg=env_cfg,
    )

    manager_env = env.unwrapped

    print("DEBUG 1: before env.reset()", flush=True)
    obs, _ = env.reset(seed=args.seed)
    print("DEBUG 2: after env.reset()", flush=True)

    policy_obs = obs["policy"]

    obs_dim = policy_obs.shape[-1]
    action_dim = env.action_space.shape[-1]

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension     : {action_dim}")

    # Policy was loaded before Isaac Sim startup.
    # Only move the already-loaded policy to CUDA here.
    policy = preloaded_policy.to(device)
    policy.eval()

    print("Checkpoint load: PASS")

    initial_rel = peg_hole_relative_position(manager_env)

    initial_xy = torch.linalg.vector_norm(
        initial_rel[:, :2],
        dim=1,
    )

    initial_z = (
        manager_env.scene["object"]
        .data.root_pos_w[:, 2]
        .clone()
    )

    episode_reward = torch.zeros(
        args.num_envs,
        device=device,
    )

    episode_steps = torch.zeros(
        args.num_envs,
        dtype=torch.long,
        device=device,
    )

    episode_min_xy = initial_xy.clone()

    episode_max_insertion = torch.zeros(
        args.num_envs,
        device=device,
    )

    episode_success = torch.zeros(
        args.num_envs,
        dtype=torch.bool,
        device=device,
    )

    completion_step = torch.zeros(
        args.num_envs,
        dtype=torch.long,
        device=device,
    )

    rewards = []
    steps = []
    xy_errors = []
    insertions = []
    successes = []
    completion_steps = []

    print("DEBUG 3: before evaluation loop", flush=True)

    for step in range(1, args.num_steps + 1):
        print(f"DEBUG 4: step {step} start", flush=True)

        with torch.inference_mode():

            policy_obs = obs["policy"]

            actions = policy.deterministic_action(
                policy_obs
            )

            if not torch.isfinite(actions).all():
                raise RuntimeError(
                    f"Non-finite action at step {step}"
                )

            obs, reward, terminated, truncated, extras = env.step(
                actions
            )

            done = terminated | truncated

            episode_reward += reward
            episode_steps += 1

            rel = peg_hole_relative_position(
                manager_env
            )

            xy_error = torch.linalg.vector_norm(
                rel[:, :2],
                dim=1,
            )

            current_z = (
                manager_env.scene["object"]
                .data.root_pos_w[:, 2]
            )

            insertion = initial_z - current_z

            episode_min_xy = torch.minimum(
                episode_min_xy,
               xy_error,
            )

            episode_max_insertion = torch.maximum(
                episode_max_insertion,
                insertion,
            )

            success_now = (
                (episode_min_xy <= XY_TOL)
                & (
                    episode_max_insertion
                    >= INSERTION_TARGET
                )
            )

            newly_successful = (
                success_now & ~episode_success
            )

            completion_step[newly_successful] = episode_steps[newly_successful]
            episode_success |= success_now

            done_ids = torch.nonzero(
                done,
                as_tuple=False,
            ).flatten()

            if done_ids.numel() > 0:

                for idx in done_ids.tolist():

                    rewards.append(
                        float(episode_reward[idx].item())
                    )

                    steps.append(
                        int(episode_steps[idx].item())
                    )

                    xy_errors.append(
                        float(episode_min_xy[idx].item())
                    )

                    insertions.append(
                        float(
                            episode_max_insertion[idx].item()
                        )
                    )

                    success = bool(
                        episode_success[idx].item()
                    )

                    successes.append(success)

                    if (
                        success
                        and completion_step[idx].item() > 0
                    ):
                        completion_steps.append(
                            int(
                                completion_step[idx].item()
                            )
                        )

                reset_rel = peg_hole_relative_position(
                    manager_env
                )

                reset_xy = torch.linalg.vector_norm(
                    reset_rel[:, :2],
                    dim=1,
              )

                reset_z = (
                    manager_env.scene["object"]
                    .data.root_pos_w[:, 2]
                )

                episode_reward[done_ids] = 0.0
                episode_steps[done_ids] = 0
                episode_min_xy[done_ids] = reset_xy[done_ids]
                episode_max_insertion[done_ids] = 0.0
                episode_success[done_ids] = False
                completion_step[done_ids] = 0
                initial_z[done_ids] = reset_z[done_ids]

        if step % 50 == 0 or step == args.num_steps:

            print(
                f"Step {step:3d}/{args.num_steps} | "
                f"XY={xy_error.mean().item() * 1000:.4f} mm | "
                f"Insertion={insertion.mean().item() * 1000:.4f} mm | "
                f"Episodes={len(rewards)} | "
                f"Success={sum(successes)}",
                flush=True,
            )

    active_success = (
        (episode_min_xy <= XY_TOL)
        & (
            episode_max_insertion
             >= INSERTION_TARGET
        )
    )

    for idx in range(args.num_envs):

        rewards.append(
            float(episode_reward[idx].item())
        )

        steps.append(
            int(episode_steps[idx].item())
        )

        xy_errors.append(
            float(episode_min_xy[idx].item())
        )

        insertions.append(
            float(episode_max_insertion[idx].item())
        )

        success = bool(active_success[idx].item())
        successes.append(success)

        if (
            success
            and completion_step[idx].item() > 0
        ):
            completion_steps.append(
                int(completion_step[idx].item())
            )

    success_rate = (
        100.0 * sum(successes) / len(successes)
    )

    mean_reward = (
        sum(rewards) / len(rewards)
    )

    mean_xy_mm = (
        sum(xy_errors)
        / len(xy_errors)
        * 1000.0
    )

    mean_insertion_mm = (
        sum(insertions)
        / len(insertions)
        * 1000.0
    )

    mean_steps = (
        sum(steps)
        / len(steps)
    )

    if completion_steps:

        mean_completion_steps = (
            sum(completion_steps)
            / len(completion_steps)
        )

        completion_time = (
            mean_completion_steps
            * float(manager_env.step_dt)
        )

    else:

        mean_completion_steps = float("nan")
        completion_time = float("nan")

    print()
    print("=" * 70)
    print("TD3 M9.5 RESULTS")
    print("=" * 70)
    print(f"Success Rate        : {success_rate:.4f} %")
    print(f"Episode Reward      : {mean_reward:.6f}")
    print(f"XY Alignment Error  : {mean_xy_mm:.6f} mm")
    print(f"Insertion Depth     : {mean_insertion_mm:.6f} mm")
    print(f"Episode Steps       : {mean_steps:.4f}")
    print(f"Completion Steps    : {mean_completion_steps:.4f}")
    print(f"Completion Time     : {completion_time:.6f} s")
    print(f"Episode Samples     : {len(rewards)}")
    print(f"Successful Episodes : {sum(successes)}")
    print("=" * 70)

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True,
    )

    with open(args.output, "w", encoding="utf-8") as f:

        f.write("TD3 M9.5 Evaluation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Num envs: {args.num_envs}\n")
        f.write(f"Num steps: {args.num_steps}\n")
        f.write(f"Success Rate: {success_rate:.8f}\n")
        f.write(f"Episode Reward: {mean_reward:.8f}\n")
        f.write(
            f"XY Alignment Error mm: "
            f"{mean_xy_mm:.8f}\n"
        )
        f.write(
            f"Insertion Depth mm: "
            f"{mean_insertion_mm:.8f}\n"
        )
        f.write(
            f"Episode Steps: "
            f"{mean_steps:.8f}\n"
        )
        f.write(
            f"Completion Steps: "
            f"{mean_completion_steps:.8f}\n"
        )
        f.write(
            f"Completion Time s: "
            f"{completion_time:.8f}\n"
        )

    env.close()
if __name__ == "__main__":

    try:
        main()
    finally:
        simulation_app.close()
