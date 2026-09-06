# M9 Training Performance Analysis — Benchmark Results

## Benchmark Protocol

- Environment: `Isaac-PegInHole-Franka-IK-Abs-v0`
- Training budget: 10,000 iterations/steps per algorithm
- Training environments: 64
- Seed: 42
- Evaluation environments: 4096
- Evaluation horizon: 250 environment steps
- Environment step: 0.02 s
- Evaluation policy: deterministic
- DDPG exploration during training: Gaussian noise N(0, 0.1)
- DDPG evaluation: no exploration noise
- Success criterion:
  - XY alignment error <= 0.5 mm
  - insertion depth >= 10 mm

## PPO — 10K

| Metric | Value |
|---|---:|
| Mean Reward — early window | 1.007866 |
| Mean Reward — late window | 0.862887 |
| XY Alignment — early | 0.468577 |
| XY Alignment — late | 0.473681 |
| Insertion Progress — early | 0.225928 |
| Insertion Progress — late | 0.228400 |
| Position Error — early | 0.825766 |
| Position Error — late | 0.561364 |
| Orientation Error — early | 2.428918 |
| Orientation Error — late | 3.046987 |
| Mean Noise Std — early | 0.998957 |
| Mean Noise Std — late | 0.071766 |

PPO showed convergence/plateau behavior, with substantial reduction in position error and policy noise, while insertion progress remained limited.

## SAC — 10K

| Metric | Value |
|---|---:|
| Success Rate | 0.0000 % |
| Episode Reward | 1.755163 |
| XY Alignment Error | 0.618824 mm |
| Insertion Depth | 1.500081 mm |
| Episode Steps | 125.0000 |
| Completion Steps | NaN |
| Completion Time | NaN s |
| Episode Samples | 8192 |
| Successful Episodes | 0 |

SAC achieved stable partial approach/alignment and shallow insertion, but did not satisfy the complete-insertion success criterion.

## DDPG — 10K SafeDDPG

| Metric | Value |
|---|---:|
| Success Rate | 0.2686 % |
| Episode Reward | 1.736759 |
| XY Alignment Error | 0.618713 mm |
| Insertion Depth | 1.634498 mm |
| Episode Steps | 125.0000 |
| Completion Steps | 152.6364 |
| Completion Time | 3.052727 s |
| Episode Samples | 8192 |
| Successful Episodes | 22 |

DDPG achieved the deepest average insertion and the highest completion success rate among SAC and DDPG under the evaluated configuration, while overall complete-insertion success remained very low.

## Comparative Summary

| Algorithm | Reward | XY Error (mm) | Insertion (mm) | Success |
|---|---:|---:|---:|---:|
| PPO | 0.862887 | 0.561364 | 0.228400 | Not reported in the PPO M9 aggregate |
| SAC | 1.755163 | 0.618824 | 1.500081 | 0.0000 % |
| DDPG | 1.736759 | 0.618713 | 1.634498 | 0.2686 % |

## M9 Conclusion

Under the current Peg-in-Hole configuration, PPO achieved the best XY alignment among the reported late-window metrics, while DDPG achieved the deepest average insertion and the highest measured completion success rate. SAC produced similar alignment and insertion behavior to DDPG but recorded no successful complete-insertion episodes. None of the evaluated algorithms demonstrated reliable complete insertion under the current task configuration.
