import random
import gym
import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from REINFORCE import REINFORCE
from DQN import DQN
from utils import standardPreprocess

# env = gym.make("Amidar-v0", render_mode="human", obs_type= "grayscale")
env = gym.make("LunarLander-v2", render_mode="human")
algo = DQN(env, batchSize=24, gamma=0.99)
algo.train(3000)
env.close()
