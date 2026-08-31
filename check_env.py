from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import carb

settings = carb.settings.get_settings()

root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

print("\n================ BEFORE ================")
print("Carbonite setting:", settings.get("/persistent/isaac/asset_root/cloud"))

settings.set("/persistent/isaac/asset_root/cloud", root)

print("\n================ AFTER ================")
print("Carbonite setting:", settings.get("/persistent/isaac/asset_root/cloud"))

from isaaclab.utils import assets

print("\n================ ISAACLAB ================")
print("NUCLEUS_ASSET_ROOT_DIR =", assets.NUCLEUS_ASSET_ROOT_DIR)
print("ISAAC_NUCLEUS_DIR      =", assets.ISAAC_NUCLEUS_DIR)
print("ISAACLAB_NUCLEUS_DIR   =", assets.ISAACLAB_NUCLEUS_DIR)

print("============================================\n")

simulation_app.close()
