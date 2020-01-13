import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMapEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNMapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, stride=2)
        self.conv2 = nn.Conv2d(128, 256, 5, stride=3)
        self.conv3 = nn.Conv2d(256, 64, 5, stride=2)
        self.fc = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc(x))
        return x
