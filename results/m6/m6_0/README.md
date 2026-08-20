# M6.0 — Peg-in-Hole Baseline Characterization

## Objective

Document the validated baseline configuration inherited from the
Isaac Lab Franka IK-Absolute Lift environment before introducing
M6 task-training and evaluation changes.

## Git Baseline

- Branch: `milestone-5-task-mdp`
- Baseline commit: `0bdaa16`
- M5.9 status: PPO end-to-end training pipeline validated

## Environment Baseline

Environment:

`Isaac-PegInHole-Franka-IK-Abs-v0`

The Peg-in-Hole environment inherits its base scene and simulation
configuration from the official Franka IK-Absolute Lift configuration.

## Ground Plane

The inherited environment contains a ground plane:

- Prim path: `/World/GroundPlane`
- Spawn configuration: `GroundPlaneCfg`
- Initial position: `(0, 0, -1.05)`

No additional ground plane is introduced in M6.0.

## Simulation Parameters

| Parameter | Value |
|---|---:|
| Physics timestep | 0.01 s |
| Physics frequency | 100 Hz |
| Decimation | 2 |
| Control timestep | 0.02 s |
| Episode length | 5.0 s |
| Default number of environments | 4096 |
| Environment spacing | 2.5 m |

## Peg Geometry

| Parameter | Value |
|---|---:|
| Peg radius | 0.010 m |
| Peg diameter | 0.020 m |
| Peg height | 0.050 m |

## Hole Geometry

| Parameter | Value |
|---|---:|
| Hole radius | 0.0105 m |
| Hole diameter | 0.021 m |

The nominal radial clearance is:

`0.0005 m = 0.5 mm`

The nominal diametral clearance is:

`0.001 m = 1.0 mm`

## Fixture Geometry

| Parameter | Value |
|---|---:|
| Fixture width | 0.080 m |
| Fixture depth | 0.080 m |
| Fixture height | 0.005 m |

The hole is represented by four rigid cuboid fixture sections,
leaving an actual opening at the center.

## M5.9 Baseline

M5.9 validated:

- Gymnasium environment registration
- Environment configuration loading
- RSL-RL PPO configuration loading
- Isaac Sim initialization
- Environment creation
- PPO actor/critic construction
- Environment rollout
- PPO optimization
- TensorBoard logging
- Model checkpoint generation

The M5.9 smoke test validated the training pipeline but did not
establish task-level Peg-in-Hole success.

## M6.0 Scope

M6.0 is a characterization and documentation milestone.

No task reward, observation, geometry, simulation, or PPO parameters
are modified in this milestone.

The purpose is to establish a reproducible baseline before
task-level training and evaluation work begins.

## Next Milestone

M6.1 will prepare the task for controlled training and evaluation,
including explicit task-success metrics and an evaluation protocol.
