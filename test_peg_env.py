import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab

print("STEP 1: Isaac Sim started", flush=True)
print("STEP 2: Creating PegInHole environment...", flush=True)

env = gym.make(
    "Isaac-PegInHole-Franka-IK-Abs-v0",
    num_envs=1,
)

print("STEP 3: ENV CREATION SUCCESS", flush=True)
print(env, flush=True)

env.close()

print("STEP 4: ENV CLOSED", flush=True)

simulation_app.close()
