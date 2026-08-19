"""M5.9 - PPO end-to-end smoke test for Peg-in-Hole."""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="M5.9 PPO end-to-end smoke test."
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
    "--max_iterations",
    type=int,
    default=2,
)

# RSL-RL arguments required by the configuration updater.
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument(
    "--logger",
    type=str,
    default=None,
    choices={"wandb", "tensorboard", "neptune"},
)
parser.add_argument("--log_project_name", type=str, default=None)

# Isaac Lab / Isaac Sim arguments.
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# ---------------------------------------------------------------------
# Launch Isaac Sim FIRST.
# ---------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------
# IMPORTANT:
# Isaac Sim is initialized now, so pxr and Isaac Lab are available.
# Importing our package registers the Peg-in-Hole Gym environment.
# ---------------------------------------------------------------------

import isaac_lab  # noqa: F401

# ---------------------------------------------------------------------
# Isaac Lab / RL imports AFTER AppLauncher.
# ---------------------------------------------------------------------

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def update_rsl_rl_cfg(
    agent_cfg: RslRlBaseRunnerCfg,
    args_cli: argparse.Namespace,
):
    """Apply the required CLI overrides to the RSL-RL configuration."""

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume

    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run

    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger

    if (
        agent_cfg.logger in {"wandb", "neptune"}
        and args_cli.log_project_name
    ):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


@hydra_task_config(
    args_cli.task,
    "rsl_rl_cfg_entry_point",
)
def main(
    env_cfg: ManagerBasedRLEnvCfg
    | DirectRLEnvCfg
    | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Run the M5.9 PPO smoke test."""

    print("=== M5.9 PPO SMOKE TEST ===")
    print(f"Environment: {args_cli.task}")
    print(f"PPO config: {agent_cfg.__class__.__name__}")

    # -------------------------------------------------------------
    # CLI overrides
    # -------------------------------------------------------------

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = (
        args_cli.num_envs
        if args_cli.num_envs is not None
        else env_cfg.scene.num_envs
    )

    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

     # -------------------------------------------------------------
    # Environment seed / device
    # -------------------------------------------------------------

    env_cfg.seed = agent_cfg.seed

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    print(f"num_envs: {env_cfg.scene.num_envs}")
    print(f"num_steps_per_env: {agent_cfg.num_steps_per_env}")
    print(f"max_iterations: {agent_cfg.max_iterations}")
    print(f"device: {env_cfg.sim.device}")

    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------

    log_root_path = os.path.abspath(
        os.path.join(
            "logs",
             "rsl_rl",
            agent_cfg.experiment_name,
        )
    )

    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir_name = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    if agent_cfg.run_name:
        log_dir_name += f"_{agent_cfg.run_name}"

    log_dir = os.path.join(
        log_root_path,
        log_dir_name,
    )

    env_cfg.log_dir = log_dir

    # -------------------------------------------------------------
    # Create environment
    # -------------------------------------------------------------

    print("[M5.9] Creating environment...")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    print("[M5.9] Environment created successfully.")

    # -------------------------------------------------------------
    # Convert multi-agent environment if required
    # -------------------------------------------------------------

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------
    # Resume support
    # -------------------------------------------------------------

    if (
        agent_cfg.resume
        or agent_cfg.algorithm.class_name == "Distillation"
    ):
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    # -------------------------------------------------------------
    # RSL-RL wrapper
    # -------------------------------------------------------------

    print("[M5.9] Creating RSL-RL vectorized wrapper...")

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    print("[M5.9] RSL-RL wrapper created successfully.")

    # -------------------------------------------------------------
    # PPO runner
    # -------------------------------------------------------------

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )

    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )

    else:
        raise ValueError(
            f"Unsupported runner class: {agent_cfg.class_name}"
        )

    print("[M5.9] PPO runner created successfully.")

    # -------------------------------------------------------------
    # Save configuration
    # -------------------------------------------------------------

    os.makedirs(
        os.path.join(log_dir, "params"),
        exist_ok=True,
    )

    dump_yaml(
        os.path.join(
            log_dir,
            "params",
            "env.yaml",
        ),
        env_cfg,
    )

    dump_yaml(
        os.path.join(
 log_dir,
            "params",
            "agent.yaml",
        ),
        agent_cfg,
    )

    # -------------------------------------------------------------
    # Checkpoint loading if requested
    # -------------------------------------------------------------

    if (
        agent_cfg.resume
        or agent_cfg.algorithm.class_name == "Distillation"
    ):
        print(
            f"[INFO] Loading model checkpoint from: {resume_path}"
        )
        runner.load(resume_path)

    # -------------------------------------------------------------
    # PPO training
    # -------------------------------------------------------------

    print("")
    print("==============================================")
    print(" M5.9 PPO END-TO-END SMOKE TEST")
    print("==============================================")
    print(
        f"Learning iterations: {agent_cfg.max_iterations}"
    )
    print(
        f"Steps per environment: {agent_cfg.num_steps_per_env}"
    )
    print(
         f"Environments: {env_cfg.scene.num_envs}"
    )
    print("==============================================")
    print("")

    start_time = time.time()

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    elapsed = time.time() - start_time

    print("")
    print(
        f"[M5.9] PPO smoke test finished in "
        f"{round(elapsed, 2)} seconds."
    )

    print("")
    print("=== M5.9 PPO SMOKE TEST PASS ===")

    # -------------------------------------------------------------
    # Close environment
    # -------------------------------------------------------------

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
