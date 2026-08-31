from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
})

from isaacsim.core.utils.nucleus import get_assets_root_path
from isaaclab.utils import assets as assets_utils

print("\n================ FRANKA ASSET PATH CHECK ================\n")

root = get_assets_root_path()

print("Nucleus root:")
print(root)

candidates = [
    f"{root}/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
    f"{root}/Isaac/Robots/FrankaEmika/panda_instanceable.usd",
    f"{root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    f"{root}/Isaac/Robots/Franka/franka.usd",
]

try:
    from omni.client import stat

    print("\n================ CANDIDATE TESTS ================\n")

    for path in candidates:
        result, entry = stat(path)
        print(f"{result!s:25} {path}")

except Exception as e:
    print("\nSTAT ERROR:")
    print(repr(e))

print("\n================ ISAACLAB CONFIG ================\n")

print("NUCLEUS_ASSET_ROOT_DIR :", assets_utils.NUCLEUS_ASSET_ROOT_DIR)
print("ISAAC_NUCLEUS_DIR      :", assets_utils.ISAAC_NUCLEUS_DIR)
print("ISAACLAB_NUCLEUS_DIR   :", assets_utils.ISAACLAB_NUCLEUS_DIR)

print("\n=================================================\n")

simulation_app.close()
