import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class ImageNet(nn.Module):
    def __init__(self, numActions):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=2592, out_features=256)
        self.output = nn.Linear(256, numActions)

    def forward(self, input):
        conv1 = torch.relu(self.conv1(input.unsqueeze(0)))
        conv2 = torch.relu(self.conv2(conv1))
        flatten = conv2.view(conv2.size(0), -1)
        fc1 = torch.relu(self.fc1(flatten))
        output = self.output(fc1).flatten()
        return output
    
class ProbRegNet(nn.Module):
    def __init__(self, numActions, FCSize, numObs):
        super(ProbRegNet, self).__init__()
        self.fc1 = nn.Linear(numObs, FCSize)
        self.softmax = nn.Linear(FCSize, numActions)

    def forward(self, input):
        fc1 = F.relu(self.fc1(input))
        output = F.softmax(self.softmax(fc1), dim=1)
        return output
    
class RegNet(nn.Module):
    def __init__(self, numActions, FCSize, numObs):
        super(RegNet, self).__init__()
        self.fc1 = nn.Linear(numObs, FCSize)
        self.fc2 = nn.Linear(FCSize, numActions)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        output = self.fc2(x)
        return output