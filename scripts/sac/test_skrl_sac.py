"""M9.1 - SKRL SAC smoke test for Peg-in-Hole."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher


# ================================================================
# CLI
# ================================================================

parser = argparse.ArgumentParser(
    description="M9.1 SKRL SAC smoke test for Peg-in-Hole."
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-PegInHole-Franka-IK-Abs-v0",
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=64,
)

parser.add_argument(
    "--max_steps",
    type=int,
    default=200,
)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Prevent Hydra from consuming our CLI arguments.
sys.argv = [sys.argv[0]] + hydra_args


# ================================================================
# Launch Isaac Sim FIRST
# ================================================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ================================================================
# Imports AFTER Isaac Sim
# ================================================================

import gymnasium as gym
import torch
import skrl

import isaac_lab  # registers the environment

from isaaclab.utils.dict import print_dict


# ================================================================
# Main
# ================================================================

def main():

    print("")
    print("================================================")
    print(" M9.1 SKRL SAC SMOKE TEST")
    print("================================================")
    print(f"Task:       {args_cli.task}")
    print(f"Num envs:   {args_cli.num_envs}")
    print(f"Max steps:  {args_cli.max_steps}")
    print(f"SKRL:       {skrl.__version__}")
    print(f"Torch:      {torch.__version__}")
    print(f"CUDA:       {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    print("================================================")
    print("")

    # ------------------------------------------------------------
    # Import the project's actual environment configuration
    # ------------------------------------------------------------

    from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg

    env_cfg = PegInHoleEnvCfg()

    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    print("[SAC] Environment configuration created.")
    print(f"[SAC] Device: {env_cfg.sim.device}")
    print(f"[SAC] Environments: {env_cfg.scene.num_envs}")

    # ------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------

    print("[SAC] Creating Peg-in-Hole environment...")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    print("[SAC] Environment created successfully.")
    print("")
    print("[SAC] Observation space:")
    print(env.observation_space)

    print("")
    print("[SAC] Action space:")
    print(env.action_space)

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------

    print("")
    print("[SAC] Resetting environment...")

    obs, info = env.reset()

    print("[SAC] Reset successful.")
    print("[SAC] Observation type:", type(obs))

    if isinstance(obs, dict):
        print("[SAC] Observation keys:", list(obs.keys()))

    # ------------------------------------------------------------
    # Action test
    # ------------------------------------------------------------

    print("")
    print("[SAC] Testing action interface...")

    action_space = env.action_space

    print("[SAC] Action space:", action_space)

    action = action_space.sample()

    print("[SAC] Sample action shape:", getattr(action, "shape", None))

    # ------------------------------------------------------------
    # Short environment rollout
    # ------------------------------------------------------------

    print("")
    print("[SAC] Running environment rollout...")

    start_time = time.time()

    for step in range(args_cli.max_steps):

        print(f"[SAC] Step {step + 1}/{args_cli.max_steps}: sampling action...", flush=True)
        action = torch.zeros((args_cli.num_envs, 8), device="cuda:0", dtype=torch.float32)
        print(f"[SAC] Step {step + 1}/{args_cli.max_steps}: using ZERO action {tuple(action.shape)}", flush=True)
        print(f"[SAC] Step {step + 1}/{args_cli.max_steps}: calling env.step()...", flush=True)
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[SAC] Step {step + 1}/{args_cli.max_steps}: env.step() returned in {time.time() - step_start:.3f}s", flush=True)

        if step == 0:
            print("[SAC] First step successful.")
            print("[SAC] Reward:", reward)

        if bool(torch.as_tensor(terminated).any()) or \
           bool(torch.as_tensor(truncated).any()):

            obs, info = env.reset()

    elapsed = time.time() - start_time

    print("")
    print(
        f"[SAC] Rollout completed: "
        f"{args_cli.max_steps} steps"
    )
    print(f"[SAC] Elapsed: {elapsed:.2f} sec")

    # ------------------------------------------------------------
    # Close
    # ------------------------------------------------------------

    env.close()

    print("")
    print("================================================")
    print(" M9.1 ENVIRONMENT / ACTION INTERFACE PASS")
    print("================================================")


if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()
