import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMapEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNMapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 7, stride=3, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 4, 5, stride=2, bias=False)
        self.fc1 = nn.Linear(4 * 7 * 7 + 2, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, v):
        x = self.pool(F.relu(self.conv1(x)))  # (94 - 7)/3 + 1 = 30 / 2 = 15
        x = self.pool(F.relu(self.conv2(x)))  # (15 - 5)/2 + 1 = 6 / 2 = 3
        x = x.view(-1, 4 * 7 * 7)
        x = torch.cat((x, v), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
