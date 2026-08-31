from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
})

import omni.client

root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
paths = [
    f"{root}/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
    f"{root}/Robots/FrankaEmika/panda_instanceable.usd",
]

print("\n================ FRANKA ASSET CHECK ================\n")

for path in paths:
    print("CHECK:", path)
    result, entries = omni.client.list(path.rsplit("/", 1)[0])

    print("RESULT:", result)

    if entries:
        for entry in entries:
            print("  ", entry.relative_path)

print("\n=====================================================\n")

simulation_app.close()
