import torch
import torch.nn as nn
import torch.nn.functional as F


class SleepScoreModel(nn.Module):
    def __init__(self, input_size=20):
        super(SleepScoreModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32, 1)  # Output is a single sleep score

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x)) * 100  # Scale output to 0–100
        return x