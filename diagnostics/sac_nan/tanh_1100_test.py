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

# -------------------------------------------------------------------------
# Diagnostic wrapper: detect the first SAC update that corrupts parameters
# -------------------------------------------------------------------------
_original_sac_update = SAC.update

def _diagnostic_sac_update(self, timestep, timesteps):
    # Check parameters before update
    for model_name, model in self.models.items():
        for param_name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                print(
                    f"[FIRST BAD BEFORE] update={timestep} "
                    f"model={model_name} param={param_name}",
                    flush=True,
                )
                raise RuntimeError("Non-finite parameter before SAC update")

    result = _original_sac_update(
        self,
        timestep=timestep,
        timesteps=timesteps,
    )

    # Check parameters after update
    for model_name, model in self.models.items():
        for param_name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                print(
                    f"[FIRST BAD AFTER] update={timestep} "
                    f"model={model_name} param={param_name}",
                    flush=True,
                )
                raise RuntimeError(
                    f"First non-finite parameter detected at update {timestep}"
                )

    if timestep >= 1000:
        print(
            f"[FINITE] update={timestep} all model parameters finite",
            flush=True,
        )

    return result

SAC.update = _diagnostic_sac_update


# =============================================================================
# Configuration
# =============================================================================

ENV_ID = "Isaac-PegInHole-Franka-IK-Abs-v0"

LOG_DIR = "./results/benchmark_v1/sac/validation_1200/logs"


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


def install_act_diagnostics(agent):
    agent._diag_timestep = -1

    original_update = SAC.update

    def diagnostic_update(self, timestep, timesteps):
        agent._diag_timestep = timestep
        return original_update(self, timestep=timestep, timesteps=timesteps)

    SAC.update = diagnostic_update

    original_policy_act = agent.policy.act
    original_c1_act = agent.critic_1.act
    original_c2_act = agent.critic_2.act

    def policy_act_diag(*args, **kwargs):
        actions, outputs = original_policy_act(*args, **kwargs)
        t = agent._diag_timestep

        if t >= 1070:
            log_prob = outputs.get("log_prob")
            print(
                f"[POLICY] update={t} "
                f"actions=({actions.detach().min().item():.3e},"
                f"{actions.detach().max().item():.3e}) "
                f"finite={torch.isfinite(actions).all().item()} "
                f"log_prob=({log_prob.detach().min().item():.3e},"
                f"{log_prob.detach().max().item():.3e}) "
                f"log_finite={torch.isfinite(log_prob).all().item()}",
                flush=True,
            )
        # TEMP TEST: bound SAC actions before environment/critic use
        actions = torch.tanh(actions)
        return actions, outputs

    def critic_diag(original_act, name):
        def wrapped(*args, **kwargs):
            result = original_act(*args, **kwargs)
            t = agent._diag_timestep

            if t >= 1070:
                q = result[0]
                inputs = args[0] if args else kwargs
                actions = inputs.get("taken_actions")

                print(
                    f"[{name}] update={t} "
                    f"Q=({q.detach().min().item():.3e},"
                    f"{q.detach().max().item():.3e}) "
                    f"Q_finite={torch.isfinite(q).all().item()} "
                    f"action=({actions.detach().min().item():.3e},"
                     f"action=({actions.detach().min().item():.3e},"
                    f"{actions.detach().max().item():.3e}) "
                    f"action_finite={torch.isfinite(actions).all().item()}",
                    flush=True,
                )
            return result
        return wrapped

    agent.policy.act = policy_act_diag
    agent.critic_1.act = critic_diag(original_c1_act, "CRITIC1")
    agent.critic_2.act = critic_diag(original_c2_act, "CRITIC2")

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
    install_act_diagnostics(agent)

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
