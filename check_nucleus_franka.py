from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
})

print("\n================ NUCLEUS / FRANKA CHECK ================\n")

# IMPORTANT: imports only AFTER SimulationApp
from omni.isaac.nucleus import get_assets_root_path
from isaaclab.utils.assets import (
    NUCLEUS_ASSET_ROOT_DIR,
    ISAAC_NUCLEUS_DIR,
    ISAACLAB_NUCLEUS_DIR,
)

print("NUCLEUS_ASSET_ROOT_DIR :", NUCLEUS_ASSET_ROOT_DIR)
print("ISAAC_NUCLEUS_DIR      :", ISAAC_NUCLEUS_DIR)
print("ISAACLAB_NUCLEUS_DIR   :", ISAACLAB_NUCLEUS_DIR)

try:
    root = get_assets_root_path()
    print("get_assets_root_path() :", root)
except Exception as e:
    print("get_assets_root_path ERROR:", repr(e))

print("\n=========================================================\n")

simulation_app.close()
