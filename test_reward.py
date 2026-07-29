from reward.reward_function import RewardFunction

reward_fn = RewardFunction()

reward = reward_fn.compute_reward(
    distance=0.05,
    success=False,
    collision=False,
)

print("Reward:", reward)