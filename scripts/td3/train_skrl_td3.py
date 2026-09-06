#!/usr/bin/env python3

import argparse
import copy

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Train Peg-in-Hole with SKRL TD3"
)

parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaac_lab  # noqa: F401
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer


ENV_ID = "Isaac-PegInHole-Franka-IK-Abs-v0"

LOG_DIR = "./results/benchmark_v1/td3/benchmark_10k/logs"


class Actor(DeterministicMixin, Model):

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
            torch.nn.Linear(self.num_observations, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, self.num_actions),
            torch.nn.Tanh(),
        )

    def compute(self, inputs, role=""):
         return self.net(inputs["observations"]), {}


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
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 1),
        )

    def compute(self, inputs, role=""):
        x = torch.cat(
            [
                inputs["observations"],
                 inputs["taken_actions"],
            ],
            dim=-1,
        )

        return self.net(x), {}


class SafeGaussianNoise:
    def __init__(self, std=0.2, device=None):
        self.std = std
        self.device = device

    def sample(self, size):
        return torch.randn(size, device=self.device) * self.std


class SafeTD3(TD3):

    def __init__(self, *args, exploration_std=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.safe_exploration_std = exploration_std

    def act(self, observations, states, *, timestep, timesteps):
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }

        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        with torch.autocast(
            device_type=self._device_type,
            enabled=self.cfg.mixed_precision,
        ):
            actions, outputs = self.policy.act(inputs, role="policy")

        noise = torch.randn_like(actions) * self.safe_exploration_std
        actions = actions + noise
        actions = torch.clamp(actions, -1.0, 1.0)
        
        self.track_data(
        "Exploration / Safe noise (max)",
        torch.max(noise).item(),
        )
        self.track_data(
        "Exploration / Safe noise (min)",
        torch.min(noise).item(),
        )
        self.track_data(
        "Exploration / Safe noise (mean)",
        torch.mean(noise).item(),
        )
        
        return actions, outputs


def main():
    print("=" * 80)
    print("SKRL TD3 - Peg-in-Hole")
    print("=" * 80)

    print(f"Environment : {ENV_ID}")
    print(f"Num envs    : {args_cli.num_envs}")
    print(f"Timesteps   : {args_cli.max_iterations}")
    print(f"Seed        : {args_cli.seed}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    print("=" * 80)

    torch.manual_seed(args_cli.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg = parse_env_cfg(
        ENV_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(
        ENV_ID,
        cfg=env_cfg,
    )

    env = wrap_env(env)

    device = env.device

    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")
    print(f"Device           : {device}")

    memory = RandomMemory(
        memory_size=100000,
        num_envs=args_cli.num_envs,
        device=device,
    )

    print("DEBUG A: before actor", flush=True)

    actor = Actor(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=False,
    )

    print("DEBUG B: after actor", flush=True)

    critic_1 = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=False,
    )

    print("DEBUG C: after critic_1", flush=True)

    critic_2 = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=False,
    )

    print("DEBUG D: after critic_2", flush=True)

    target_actor = copy.deepcopy(actor)
    target_critic_1 = copy.deepcopy(critic_1)
    target_critic_2 = copy.deepcopy(critic_2)

    models = {
        "policy": actor,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_policy": target_actor,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }

    print("DEBUG E: after target models", flush=True)

    td3_cfg = TD3_CFG(
        gradient_steps=1,
        batch_size=64,
        discount_factor=0.99,
        polyak=0.005,
        learning_rate=3e-4,
        random_timesteps=100,
        learning_starts=100,
        grad_norm_clip=1.0,
        exploration_noise=None,
        policy_delay=2,
        smooth_regularization_noise=SafeGaussianNoise,
        smooth_regularization_clip=0.5,
        experiment={
           "directory": LOG_DIR,
            "experiment_name": "peg_in_hole_td3",
            "write_interval": 10,
            "checkpoint_interval": 1000,
        },
    )

    print("DEBUG: before SafeTD3 construction", flush=True)
    print("DEBUG F: after TD3_CFG", flush=True)

    agent = SafeTD3(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=td3_cfg,
    )

    print("DEBUG: after SafeTD3 construction", flush=True)

    trainer_cfg = {
        "timesteps": args_cli.max_iterations,
        "headless": args_cli.headless,
    }

    print("DEBUG: before SequentialTrainer", flush=True)

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=trainer_cfg,
    )

    print("DEBUG: after SequentialTrainer", flush=True)

    print()
    print("=" * 80)
    print("Starting TD3 run")
    print("=" * 80)
    print()

    trainer.train()

    print()
    print("=" * 80)
    print("TD3 run finished")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
