#!/usr/bin/env python3

"""
SKRL SAC training for:
    Isaac-PegInHole-Franka-IK-Abs-v0

SKRL:
    2.1.0

Initial run:
    Short validation run to verify environment stepping,
    replay memory, SAC updates, logging and checkpoints.

The IsaacLab task configuration is intentionally unchanged.
"""

import argparse
import copy

from isaaclab.app import AppLauncher


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Train Peg-in-Hole with SKRL SAC"
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=64,
    help="Number of parallel environments.",
)

parser.add_argument(
    "--max_iterations",
    type=int,
    default=100,
    help="Number of environment steps for the validation run.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# Imports after Isaac Sim startup
# =============================================================================

import isaac_lab  # noqa: F401
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer


# =============================================================================
# Configuration
# =============================================================================

ENV_ID = "Isaac-PegInHole-Franka-IK-Abs-v0"

LOG_DIR = "./results/benchmark_v1/sac/benchmark_10k_squashed/logs"


# =============================================================================
# Models
# =============================================================================

class Policy(GaussianMixin, Model):

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
    ):
        print("[DEBUG] Policy: before Model.__init__", flush=True)
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        print("[DEBUG] Policy: before GaussianMixin.__init__", flush=True)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=False,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            reduction="sum",
        )

        print("[DEBUG] Policy: before network", flush=True)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
        )

        print("[DEBUG] Policy: network OK", flush=True)
        self.mean_layer = torch.nn.Linear(
            256,
            self.num_actions,
        )

        print("[DEBUG] Policy: mean_layer OK", flush=True)
        self.log_std_parameter = torch.nn.Parameter(
            torch.zeros(
                self.num_actions,
                device=device,
            )
        )
        print("[DEBUG] Policy: log_std_parameter OK", flush=True)
        print("[DEBUG] Policy: initialization complete", flush=True)

    def compute(self, inputs, role):

        x = self.net(inputs["observations"])

        mean = self.mean_layer(x)

        log_std = self.log_std_parameter.expand_as(mean)

        return mean, {"log_std": log_std}

    def act(self, inputs, *, role=""):
        # Squashed Gaussian policy for numerically stable SAC.
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        # Match GaussianMixin log-std bounds.
        log_std = torch.clamp(log_std, min=-20, max=2)
        outputs["log_std"] = log_std

        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean_actions, std)

         # Reparameterized Gaussian sample.
        pre_tanh_actions = distribution.rsample()

        # Bound actions to [-1, 1].
        actions = torch.tanh(pre_tanh_actions)

        # SAC log-probability with tanh Jacobian correction.
        log_prob = distribution.log_prob(pre_tanh_actions)
        log_prob = log_prob - torch.log(
            1.0 - actions.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(dim=-1, keepdim=True)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = torch.tanh(mean_actions)

        # Keep the current distribution available for SKRL.
        self._g_distribution = distribution

        return actions, outputs


class Critic(DeterministicMixin, Model):

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        DeterministicMixin.__init__(
            self,
            clip_actions=clip_actions,
        )

        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                self.num_observations + self.num_actions,
            256,
            ),
            torch.nn.ELU(),

            torch.nn.Linear(
                256,
                256,
            ),
            torch.nn.ELU(),

            torch.nn.Linear(
                256,
                256,
            ),
            torch.nn.ELU(),

            torch.nn.Linear(
                256,
                1,
            ),
        )

    def compute(self, inputs, role):

        x = torch.cat(
            [
                inputs["observations"],
                inputs["taken_actions"],
            ],
            dim=-1,
        )

        return self.net(x), {}


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 80)
    print("SKRL SAC - Peg-in-Hole")
    print("=" * 80)

    print(f"Environment : {ENV_ID}")
    print(f"Num envs    : {args_cli.num_envs}")
    print(f"Timesteps   : {args_cli.max_iterations}")
    print(f"Seed        : {args_cli.seed}")

    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 80)

    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------

    torch.manual_seed(args_cli.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    print("[DEBUG] main(): before parse_env_cfg()", flush=True)

    env_cfg = parse_env_cfg(
        ENV_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    print("[DEBUG] main(): parse_env_cfg() OK", flush=True)
    print("[DEBUG] main(): before gym.make()", flush=True)

    env = gym.make(
        ENV_ID,
        cfg=env_cfg,
    )

    print("[DEBUG] main(): gym.make() OK", flush=True)

    env = wrap_env(env)

    device = env.device

    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")

    print(
        f"Device           : "
        f"{device}"
    )

    # -------------------------------------------------------------------------
    # Replay memory
    # -------------------------------------------------------------------------

    memory = RandomMemory(
        memory_size=100000,
        num_envs=args_cli.num_envs,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Create models
    # -------------------------------------------------------------------------

    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=False,
    )

    print("[DEBUG] before Critic 1", flush=True)
    critic_1 = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=False,
    )

    print("[DEBUG] Critic 1 OK / before Critic 2", flush=True)
    critic_2 = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=False,
    )

    # Target critics must exist explicitly for SAC.
    print("[DEBUG] Critic 2 OK / before target critics", flush=True)
    target_critic_1 = copy.deepcopy(critic_1)
    target_critic_2 = copy.deepcopy(critic_2)
    print("[DEBUG] all critics/models OK", flush=True)

    models = {
        "policy": policy,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }

    # -------------------------------------------------------------------------
    # SAC configuration
    # -------------------------------------------------------------------------

    sac_cfg = SAC_CFG(
        gradient_steps=1,
        batch_size=256,

        discount_factor=0.99,

        polyak=0.005,

        learning_rate=3e-4,

        random_timesteps=1000,
        learning_starts=1000,

        grad_norm_clip=1.0,

        learn_entropy=True,

        initial_entropy_value=0.2,

        target_entropy=-float(
            env.action_space.shape[-1]
        ),

     experiment={
            "directory": LOG_DIR,
            "experiment_name": "peg_in_hole_sac",
            "write_interval": 10,
            "checkpoint_interval": 100,
        },
    )

    # -------------------------------------------------------------------------
    # SAC agent
    # -------------------------------------------------------------------------

    print("[DEBUG] before SAC()", flush=True)
    agent = SAC(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=sac_cfg,
    )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------

    trainer_cfg = {
        "timesteps": args_cli.max_iterations,
        "headless": args_cli.headless,
    }

    print("[DEBUG] SAC() OK / before SequentialTrainer()", flush=True)
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=trainer_cfg,
    )

     # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Starting SAC validation run")
    print("=" * 80)
    print()

    trainer.train()

    print()
    print("=" * 80)
    print("SAC validation run finished")
    print("=" * 80)

    env.close()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()
