# Peg-in-Hole Environment Design

## Robot
- Franka Panda (7-DOF)

## Task
The robot must insert a cylindrical peg into a hole using Cartesian control.

---

## Observation Space

| Component | Dimension |
|-----------|-----------|
| Joint Positions | 7 |
| Joint Velocities | 7 |
| End-Effector Position | 3 |
| Relative Peg-to-Hole Position | 3 |

Total Observation Dimension = **20**

---
## Action Space

Cartesian Control: 

- ΔX
- ΔY
- ΔZ
- ΔRoll
- ΔPitch
- ΔYaw

Total Action Dimension = **6**

---

## Episode Termination

### Success Condition
- Peg fully inserted into the hole.

### Failure Condition
- Maximum episode length reached.
- Robot leaves workspace.
- Unsafe collision

---
### Reward components
- Distance Reward
- Alignment Reward
- Insertion Reward
- Time Penalty
- Collision Penalty
- Success Bonus

---
