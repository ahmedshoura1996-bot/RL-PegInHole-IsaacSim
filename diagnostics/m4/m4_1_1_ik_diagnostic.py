from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaac_lab.peg_in_hole_env_cfg import PegInHoleEnvCfg


print("========================================")
print("       M4.1.1 IK DIAGNOSTIC TEST")
print("========================================")


cfg = PegInHoleEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 2.0

env = ManagerBasedRLEnv(cfg)

robot = env.scene["robot"]

ee_index = robot.body_names.index("panda_hand")


print("")
print("ROBOT / CONTROL")
print("----------------------------------------")
print("Bodies:", robot.num_bodies)
print("Joints:", robot.num_joints)
print("EE index:", ee_index)
print("EE body:", robot.body_names[ee_index])

print("")
print("ACTION MANAGER")
print("----------------------------------------")
print("Action shape:", env.action_manager.action.shape)
print(env.action_manager)


env.reset()


# ------------------------------------------------------------
# SETTLE ROBOT
# ------------------------------------------------------------

zero_action = env.action_manager.action.clone()
zero_action[:] = 0.0

for _ in range(50):
    env.step(zero_action)

env.scene.update(env.sim.get_physics_dt())


# ------------------------------------------------------------
# INITIAL STATE
# ------------------------------------------------------------

ee0 = robot.data.body_pos_w[0, ee_index].clone()
quat0 = robot.data.body_quat_w[0, ee_index].clone()
q0 = robot.data.joint_pos[0].clone()


print("")
print("INITIAL STATE")
print("----------------------------------------")

print(
    f"EE position: "
    f"X={ee0[0].item():.6f} "
    f"Y={ee0[1].item():.6f} "
    f"Z={ee0[2].item():.6f}"
)

print("EE quaternion:")
print(quat0)

print("Joint positions:")
print(q0)


# ------------------------------------------------------------
# TEST 1: HOLD CURRENT POSE
# ------------------------------------------------------------

print("")
print("TEST 1: HOLD CURRENT EE POSE")
print("----------------------------------------")

hold_action = zero_action.clone()

hold_action[0, 0:3] = ee0
hold_action[0, 3:7] = quat0
hold_action[0, 7] = 0.0

print("Command:")
print(hold_action[0])


for i in range(50):
    env.step(hold_action)
    env.scene.update(env.sim.get_physics_dt())

    if i in [0, 9, 24, 49]:
        ee = robot.data.body_pos_w[0, ee_index]

        print(
            f"STEP {i + 1:02d} | "
            f"X={ee[0].item():.6f} "
            f"Y={ee[1].item():.6f} "
            f"Z={ee[2].item():.6f}"
        )


ee_hold = robot.data.body_pos_w[0, ee_index].clone()


print("")
print("HOLD ERROR")
print("----------------------------------------")

print(
    f"dX={(ee_hold[0] - ee0[0]).item():.6f} m"
)
print(
    f"dY={(ee_hold[1] - ee0[1]).item():.6f} m"
)
print(
    f"dZ={(ee_hold[2] - ee0[2]).item():.6f} m"
)


# ------------------------------------------------------------
# TEST 2: SMALL +Z COMMAND
# ------------------------------------------------------------

print("")
print("TEST 2: +Z 0.01 m")
print("----------------------------------------")

target_z = ee_hold[2] + 0.01

action_z = zero_action.clone()

action_z[0, 0] = ee_hold[0]
action_z[0, 1] = ee_hold[1]
action_z[0, 2] = target_z

action_z[0, 3:7] = quat0
action_z[0, 7] = 0.0


print(
    f"Target: "
    f"X={action_z[0,0].item():.6f} "
    f"Y={action_z[0,1].item():.6f} "
    f"Z={action_z[0,2].item():.6f}"
)


for i in range(50):
    env.step(action_z)
    env.scene.update(env.sim.get_physics_dt())

    if i in [0, 1, 4, 9, 24, 49]:
        ee = robot.data.body_pos_w[0, ee_index]

        print(
            f"STEP {i + 1:02d} | "
            f"X={ee[0].item():.6f} "
            f"Y={ee[1].item():.6f} "
            f"Z={ee[2].item():.6f}"
        )


ee_z = robot.data.body_pos_w[0, ee_index].clone()


print("")
print("Z MOTION RESULT")
print("----------------------------------------")

print(
    f"Target dZ = {(target_z - ee_hold[2]).item():.6f} m"
)

print(
    f"Actual dX = {(ee_z[0] - ee_hold[0]).item():.6f} m"
)

print(
    f"Actual dY = {(ee_z[1] - ee_hold[1]).item():.6f} m"
)

print(
    f"Actual dZ = {(ee_z[2] - ee_hold[2]).item():.6f} m"
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("")
print("========================================")
print("       M4.1.1 DIAGNOSTIC COMPLETE")
print("========================================")


env.close()
simulation_app.close()
