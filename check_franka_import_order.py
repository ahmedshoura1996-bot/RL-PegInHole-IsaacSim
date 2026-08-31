from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
})

import carb

NUCLEUS_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

print("\n================ IMPORT ORDER TEST ================\n")

settings = carb.settings.get_settings()

print("BEFORE:")
print(
    "Carbonite:",
    settings.get("/persistent/isaac/asset_root/cloud")
)

print("\nSETTING CARBONITE...")
settings.set(
    "/persistent/isaac/asset_root/cloud",
    NUCLEUS_URL,
)

print("AFTER:")
print(
    "Carbonite:",
    settings.get("/persistent/isaac/asset_root/cloud")
)

print("\nIMPORTING ISAAC LAB ASSETS...")

from isaaclab.utils import assets

print("\nISAAC LAB VALUES:")
print("NUCLEUS_ASSET_ROOT_DIR =", assets.NUCLEUS_ASSET_ROOT_DIR)
print("ISAAC_NUCLEUS_DIR      =", assets.ISAAC_NUCLEUS_DIR)
print("ISAACLAB_NUCLEUS_DIR   =", assets.ISAACLAB_NUCLEUS_DIR)

print("\nIMPORTING FRANKA...")

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

print("\nFRANKA USD PATH:")
print(FRANKA_PANDA_CFG.spawn.usd_path)

print("\n====================================================\n")

simulation_app.close()
