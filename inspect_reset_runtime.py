import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab


print("========== M3.1 RESET RUNTIME CHECK ==========", flush=True)

env = gym.make(
    "Isaac-PegInHole-Franka-IK-Abs-v0",
    num_envs=1,
)

base_env = env.unwrapped

print("\nEnvironment:", flush=True)
print(type(base_env), flush=True)

print("\nEvent manager:", flush=True)
print(type(base_env.event_manager), flush=True)

print("\nActive event terms:", flush=True)
print(base_env.event_manager.active_terms, flush=True)

print("\nEvent manager public attributes:", flush=True)
print(
    [
        x
        for x in dir(base_env.event_manager)
        if not x.startswith("_")
    ],
    flush=True,
)

print("\nScene rigid objects:", flush=True)
print(base_env.scene.rigid_objects.keys(), flush=True)

print("\nPeg position BEFORE reset:", flush=True)
print(base_env.scene["object"].data.root_pos_w, flush=True)

obs, info = base_env.reset()

print("\nPeg position AFTER reset:", flush=True)
print(base_env.scene["object"].data.root_pos_w, flush=True)

print("\n========== M3.1 CHECK COMPLETE ==========", flush=True)

env.close()
simulation_app.close()
