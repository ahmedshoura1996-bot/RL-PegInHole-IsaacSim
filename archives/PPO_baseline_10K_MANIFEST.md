# PPO Peg-in-Hole Baseline

Date: 2026-08-31

## Training

Algorithm: PPO
Environment: Isaac-PegInHole-Franka-IK-Abs-v0
Environments: 64
Seed: 42
Iterations: 10,000
Steps per environment: 24
Checkpoint interval: 500

## PPO Configuration

Actor:
[256, 128, 64]

Critic:
[256, 128, 64]

Activation:
ELU

Learning rate:
1e-4

Entropy coefficient:
0.006

Gamma:
0.98

Lambda:
0.95

Learning epochs:
5

Mini-batches:
4

Clip parameter:
0.2

## Final Training Metrics

Mean reward:
0.845259

Mean episode length:
250

XY alignment reward:
0.474661

Insertion progress reward:
0.228703

Position error:
0.572994

Orientation error:
3.064045

Mean noise std:
0.070723

FPS:
1498

## 500-Iteration Analysis

First 500 iterations:
Mean reward = 1.007866
Position error = 0.825766
Insertion progress = 0.225928

Last 500 iterations:
Mean reward = 0.862887
Position error = 0.561364
Insertion progress = 0.228400

## Main Findings

1. PPO learned to reduce position error by approximately 32%.
2. XY alignment improved only approximately 1.09%.
3. Insertion progress improved only approximately 1.09%.
4. Orientation error increased by approximately 25.45%.
5. Policy exploration/noise decreased substantially.
6. Insertion-related metrics reached an apparent plateau early in training.
7. PPO is therefore retained as the baseline but is not considered the final solution.

## Next Experiment

SAC will be evaluated using the same environment, observations,
actions and reward structure wherever possible.

The objective is a fair PPO vs SAC comparison.
