import random
import gym
import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from REINFORCE import REINFORCE


env = gym.make("CartPole", render_mode="human")
algo = REINFORCE(env, epsilon=0.9, FDLayerSize=64, gamma=0.99)
mean, std_dev = algo.train(numEpisodes=200)
print(f"Mean Reward: {mean:.2f}, Standard Deviation: {std_dev:.2f}")


# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#    print(observation)
#    if terminated or truncated:
#       observation, info = env.reset()

env.close()
