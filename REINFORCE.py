from cmath import exp
from collections import deque
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sampleModels import ProbRegNet

class REINFORCE():

    def __init__(self, env, FDLayerSize=7, epsilon=0.9, gamma=0.9, alpha=0.001, decayRate=0.005):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.decayRate = decayRate
        self.net = ProbRegNet(env.action_space.n, FDLayerSize, env.observation_space.shape[0])
        self.optimizer = optim.Adam(self.net.parameters(), alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def randAct(self, obs):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        probs = self.net.forward(obs_tensor).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def actMax(self, obs):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        probs = self.net.forward(obs_tensor).cpu()
        m = Categorical(probs)
        action = np.argmax(probs.detach().numpy())
        log_prob = m.log_prob(torch.tensor(action))
        return action, log_prob
    
    def train(self, numEpisodes=3, maxTrainingSteps=1000):
        episodeRewards = []
        for episodeNum in range(numEpisodes):
            rewards, logProbs, totalRewards = self.collectEpisode(episodeNum, maxTrainingSteps)
            returns = deque(maxlen=maxTrainingSteps) 
            for t in range(len(rewards))[::-1]:
                discReturnT = (returns[0] if len(returns)>0 else 0)
                returns.appendleft(self.gamma*discReturnT + rewards[t])
            
            returns = torch.tensor(returns)
            
            policyLoss = []
            for logProb, discReturn in zip(logProbs, returns):
                policyLoss.append(-logProb * discReturn)
            policyLoss = torch.cat(policyLoss).sum()
            
            self.optimizer.zero_grad()
            policyLoss.backward()
            self.optimizer.step()
            meanReward = np.mean(episodeRewards)
            stdDev = np.std(episodeRewards)
            episodeRewards.append(totalRewards)
            print("Episode {}: Mean Reward: {:.2f}, Standard Deviation: {:.2f}".format(episodeNum, meanReward, stdDev))

        return np.mean(episodeRewards), np.std(episodeRewards)

    def collectEpisode(self, episodeNumber, maxTrainingSteps=1000):
        obs = self.env.reset()[0]
        rewards = []
        logProbs = []
        newEpsilon = (self.epsilon * exp(-self.decayRate * episodeNumber)).real
        totalRewards = 0
        for _ in range(maxTrainingSteps):
            if np.random.uniform() <= newEpsilon:
                # do the best step
                # will need to encode obs if nessecary (ex. true/false values)
                action, log_prob = self.randAct(obs)
            else:
                action, log_prob = self.actMax(obs)
            
            logProbs.append(log_prob)
            obs, reward, done, truncated, info = self.env.step(action)
            totalRewards += reward
            rewards.append(reward)

            if done:
                break
        return rewards, logProbs, totalRewards
    


            

                 


        
# net = Net(4, 7, 2)
# input = torch.randn(4)
# out = net(input)
# print(out)

