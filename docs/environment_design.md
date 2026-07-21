# Peg-in-Hole Environment Design

## Robot
- Franka Panda (7-DOF)

## Task
- Insert a cylindrical peg into a cylindrical hole.

## Observation Space
- Joint positions
- Joint velocities
- End-effector position
- End-effector orientation
- Peg position
- Hole position

## Action Space
- ΔX
- ΔY
- ΔZ
- ΔRoll
- ΔPitch
- ΔYaw

## Reward
- Distance reward
- Alignment reward
- Insertion reward
- Success bonus

## Success Condition
- Peg fully inserted into the hole.

## Failure Condition
- Maximum episode length reached.
- Robot exceeds workspace limits.