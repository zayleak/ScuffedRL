from cmath import exp
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sampleModels import ImageNet, RegNet
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, bufferSize):
        self.buffer = deque(maxlen=bufferSize)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        if len(self) < batchSize:
            return self.buffer
        return random.sample(self.buffer, batchSize)

    def __len__(self):
        return len(self.buffer)
    
class FrameBuffer:
    def __init__(self, preproccess, numFrames=4):
        self.numFrames = numFrames
        self.preproccess = preproccess
        self.frames = []

    def addFrame(self, frame):
        preprocessedFrame = self.preproccess(frame)
        self.frames.append(preprocessedFrame)
        
        if len(self.frames) > self.numFrames:
            self.frames.pop(0)

        return self.getInput()

    def __len__(self):
        return len(self.frames)

    def getInput(self):
        return np.stack(self.frames, axis=0)    

class DQN():

    def __init__(self, env, isImageObs=False, FDLayerSize=64, preproccess= lambda x: x, replayCapacity=10000, gamma=0.9, epsilon=0.9, batchSize=16, alpha=0.001, decayRate=0.005):
        self.epsilon = epsilon
        self.env = env
        self.isImageObs = isImageObs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.replayBuffer = ReplayBuffer(replayCapacity)
        self.batchSize = batchSize
        self.gamma = gamma
        self.decayRate = decayRate
        
        if isImageObs:
            self.net = ImageNet(env.action_space.n)
            self.frameBuffer = FrameBuffer(preproccess)
            self.preproccess = self.frameBuffer.addFrame
        else:
            self.net = RegNet(env.action_space.n, FDLayerSize, env.observation_space.shape[0])
            self.preproccess = preproccess

        self.optimizer = optim.Adam(self.net.parameters(), alpha)

    def randAct(self):
        return np.random.randint(0, self.env.action_space.n)
    
    def train(self, numEpisodes=10, maxTrainingSteps=1000):
        for episodeNumber in range(numEpisodes):
            meanReward, stdDev, meanLoss = self.collectEpisode(episodeNumber, maxTrainingSteps)
            print("Episode {}: Mean Reward: {:.2f}, Standard Deviation: {:.2f}, Mean Loss: {:.2f}".format(episodeNumber, meanReward, stdDev, meanLoss))

    def getQValues(self, procObs, withGradient=False):
        obsTensor = torch.tensor(procObs, dtype=torch.float32)
        if withGradient:
            return self.net(obsTensor)
        else: 
            return self.net(obsTensor).cpu().detach().numpy() 

    def actMax(self, procObs):
        return np.argmax(self.getQValues(procObs))

    def collectEpisode(self, episodeNumber, maxTrainingSteps=1000):
        obs = self.env.reset()[0]
        phiT = self.preproccess(obs)
        newEpsilon = (self.epsilon * exp(-self.decayRate * episodeNumber)).real
        episodeRewards = []
        squaredErrors = []
        for trainingStep in range(maxTrainingSteps):
            if np.random.uniform() <= newEpsilon or (trainingStep <= 4 and self.isImageObs): 
                action = self.randAct()
            else:
                action = self.actMax(phiT)
            
            obs, reward, done, truncated, info = self.env.step(action)
            phiT1 = self.preproccess(obs)

            if not self.isImageObs or trainingStep >= 5:
                self.replayBuffer.add([phiT, action, reward, phiT1, done])

            samples = self.replayBuffer.sample(self.batchSize)
            
            squaredErrors = []

            if not self.isImageObs or trainingStep >= 5:
                for sample in samples:
                    # Process each sample
                    phiJ, action, reward, phiJ1, doneSample = sample
                    if doneSample:
                        yJ = reward
                    else:
                        yJ = reward + self.gamma * torch.max(self.getQValues(phiJ1, True))

                    squaredError = torch.square(yJ - self.getQValues(phiJ, True)[action])
                    self.optimizer.zero_grad()
                    squaredError.backward()
                    self.optimizer.step()
                    squaredErrors.append(squaredError.detach())


            episodeRewards.append(reward)
            phiT = phiT1
            
            if done:
                break
        
        return np.mean(episodeRewards), np.std(episodeRewards), np.mean(squaredErrors)

            
            










        




