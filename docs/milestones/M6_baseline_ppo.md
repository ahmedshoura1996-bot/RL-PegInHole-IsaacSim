# M6 — Baseline PPO Training

## Purpose

Run the first real PPO training campaign for the Peg-in-Hole task and establish a baseline policy, checkpoints, evaluation metrics, and failure characterization.

This milestone follows the official project roadmap, where M6 is defined as **Baseline PPO Training**.

---

## M6.1 — Baseline Training Setup

### Environment

- Task: `Isaac-PegInHole-Franka-IK-Abs-v0`
- Simulator: NVIDIA Isaac Sim / Isaac Lab
- RL framework: RSL-RL
- Algorithm: PPO
- Parallel environments: 4096
- Observation dimension: 36
- Action dimension: 8
- Episode length: 250 steps

### Observation Space

The policy observation group contains:

- Joint position: 9
- Joint velocity: 9
- Object position: 3
- Target object position/orientation: 7
- Previous actions: 8

Total:

`36 observations`

### Action Space

- Arm action: 7
- Gripper action: 1

Total:

`8 actions`

---

## M6.2 — Baseline PPO Training

A real PPO training campaign was executed using 4096 parallel environments.

Training completed successfully for 100 iterations and generated RSL-RL checkpoints.

The training pipeline was verified end-to-end:

`Environment → Observations → PPO → Actions → Rewards → Runner → Checkpoints`

The PPO training smoke test from M5.9 had already passed before the M6 baseline campaign.

---

## M6.3 — Baseline Policy Evaluation

The trained policy was evaluated over 4096 parallel environments.

Evaluation metrics included:

- Success rate
- XY alignment
- Insertion depth
- Episode completion
- Mean action magnitude

### Baseline Evaluation Result

The baseline policy consistently achieved XY alignment but failed to perform insertion.

Observed behavior:

- XY alignment rate: **1.000000**
- XY aligned environments: **4096 / 4096**
- Insertion rate: **0.000000**
- Inserted environments: **0 / 4096**
- Success rate: **0.000000**
- Successful environments: **0 / 4096**
- Best observed insertion depth: approximately **3 mm**
- Episode completion rate: **1.000000**

---

## M6.4 — Policy Behavior Analysis

The policy behavior was analyzed to determine the source of failure.

The resulting behavior was classified as:

**XY ALIGNMENT SUCCESS / INSERTION FAILURE**

The policy learned to move the peg toward the hole center in the XY plane, but did not learn the subsequent downward insertion behavior.

This establishes that the main remaining difficulty is not lateral alignment but the transition from alignment to insertion.

---

## Post-M6 Diagnostic Experiments

The following experiments were conducted after the baseline result to investigate the insertion failure.

These are recorded as **diagnostic experiments**, not official roadmap milestones.

### Diagnostic Experiment A — Dense Insertion Reward

The insertion reward was further conditioned on XY alignment using a smooth Gaussian-style alignment gate.

The purpose was to encourage insertion only when the peg was sufficiently close to the hole center.

Result:

- XY alignment remained successful.
- Insertion remained unsuccessful.
- Success rate remained 0%.

Conclusion:

Alignment-gated insertion reward did not produce successful insertion with the tested training configuration.

---

## M6 Main Finding

The baseline PPO policy successfully learned the lateral alignment component of the Peg-in-Hole task but failed to learn the insertion phase.

### Final Classification

**XY Alignment Success / Insertion Failure**

This result provides a clear baseline failure mode for subsequent task development and experimentation.

---

## Checkpoints

Baseline and diagnostic training campaigns generated RSL-RL checkpoints under:

`logs/rsl_rl/peg_in_hole/`

Generated checkpoints are intentionally not committed to the Git repository.

---

## Scope Note

M6 is the official roadmap milestone for **Baseline PPO Training**.

The dense-reward and alignment-gated experiments are documented as post-M6 diagnostic experiments and are not treated as official M6.x roadmap milestones.

The next official roadmap milestone after M6 is:

**M7 — Peg-in-Hole Task Variations**

