import argparse
import inspect

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_lab

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject


print("========== ISAAC LAB API CHECK ==========", flush=True)

print("ManagerBasedRLEnv module:")
print(inspect.getfile(ManagerBasedRLEnv), flush=True)

print("\nManagerBasedRLEnv.reset:")
print(inspect.signature(ManagerBasedRLEnv.reset), flush=True)

print("\nManagerBasedRLEnv.step:")
print(inspect.signature(ManagerBasedRLEnv.step), flush=True)

print("\nRigidObject module:")
print(inspect.getfile(RigidObject), flush=True)

print("\nCreating PegInHole environment...", flush=True)

env = gym.make(
    "Isaac-PegInHole-Franka-IK-Abs-v0",
     num_envs=1,
)

print("\nEnvironment created.", flush=True)

base_env = env.unwrapped

print("\nEnvironment type:")
print(type(base_env), flush=True)

print("\nScene type:")
print(type(base_env.scene), flush=True)

print("\nScene attributes containing 'object':")
print([x for x in dir(base_env.scene) if "object" in x.lower()], flush=True)

print("\nScene attributes containing 'fixture':")
print([x for x in dir(base_env.scene) if "fixture" in x.lower()], flush=True)

print("\nTrying scene['object'] ...")
try:
    peg = base_env.scene["object"]

    print("SUCCESS: scene['object']", flush=True)
    print("Type:", type(peg), flush=True)

    print("\nPeg data type:")
    print(type(peg.data), flush=True)

    print("\nPeg root_pos_w:")
    print(peg.data.root_pos_w, flush=True)

except Exception as e:
    print("FAILED:", repr(e), flush=True)

env.close()
simulation_app.close()

print("\n========== CHECK COMPLETE ==========", flush=True)
