# Benchmark v1 — PPO vs SAC vs DDPG

## Purpose

Establish a controlled algorithm comparison for the
Peg-in-Hole task using identical environment, task,
observation, action, reward, physics, and evaluation conditions.

---

## Environment

Task:
`Isaac-PegInHole-Franka-IK-Abs-v0`

Parallel environments:
`4096`

Episode length:
`5.0 s`

Control timestep:
`0.02 s`

Episode steps:
`250`

Simulation decimation:
`2`

Physics frequency:
`100 Hz`

---

## Task Geometry

Peg:
- Diameter: `0.020 m`
- Radius: `0.010 m`
- Height: `0.050 m`

Hole:
- Diameter: `0.021 m`
- Radius: `0.0105 m`

Nominal radial clearance:
`0.0005 m`

---

## Observation Space

Observation dimension:
`36`

Observation terms:
- Joint position: 9
- Joint velocity: 9
- Peg-hole relative position: 3
- Target object pose: 7
- Previous action: 8

Total:
`36`

Observation corruption:
`enabled`

No algorithm-specific observation modification is allowed.

---

## Action Space

Total action dimension:
`8`

Arm:
`7`

Gripper:
`1`

Arm controller:
`DifferentialInverseKinematicsAction`

IK mode:
`Absolute`

IK method:
`DLS`

IK lambda:
`0.01`

Relative mode:
`False`

---

## Reward

The same environment reward is used for all algorithms.

Task rewards:
- XY alignment reward: weight `0.5`
- insertion progress reward: weight `0.5`

Regularization:
- action rate: `-0.0001`
- joint velocity: `-0.0001`

No algorithm-specific reward modification is permitted.

---

## Termination

Timeout:
`5.0 s`

Object dropping:
minimum height `-0.05 m`

No success-based early termination.

---

## Success Metric

A successful Peg-in-Hole episode requires:

1. XY error <= `0.0005 m`
2. insertion depth >= `0.010 m`

Binary success:

`XY_OK AND INSERTION_OK`

The success metric does not terminate the environment.

---

## Evaluation Protocol

For each trained algorithm:

- Environments: `4096`
- Evaluation horizon: `250` control steps
- Same environment configuration
- Same task-success metric
- Deterministic inference
- No exploration noise
- No training updates during evaluation

Primary metric:
`Success Rate`

Secondary metrics:
- XY alignment rate
- insertion rate
- insertion depth
- episode completion rate

---

## Training Budget

The historical PPO baseline run used:

- 4096 environments
- 24 steps/environment/iteration
- 10 iterations

Total environment transitions:

`4096 × 24 × 10 = 983,040`

This historical run is retained as the existing PPO baseline.

For future algorithm comparisons, the training budget must be explicitly
reported in environment transitions rather than only optimizer iterations.

---

## Reproducibility

Random seed:
`42`

All algorithm-specific hyperparameters must be recorded separately.

Environment and evaluation parameters must remain fixed.

---

## Important Note

The historical M6 documentation states 100 PPO iterations,
but the preserved run metadata in:

`logs/rsl_rl/peg_in_hole/2026-08-19_13-34-31/params/agent.yaml`

records:

`max_iterations: 10`

Therefore the preserved checkpoint run is treated as a
10-iteration / 983,040-transition PPO baseline unless
independent evidence establishes a separate 100-iteration run.
