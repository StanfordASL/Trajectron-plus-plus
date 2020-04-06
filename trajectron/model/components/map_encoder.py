import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMapEncoder(nn.Module):
    def __init__(self, map_channels, hidden_channels, output_size, masks, strides, patch_size):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        patch_size_x = patch_size[0] + patch_size[2]
        patch_size_y = patch_size[1] + patch_size[3]
        input_size = (map_channels, patch_size_x, patch_size_y)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float('nan'))

        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(nn.Conv2d(map_channels if i == 0 else hidden_channels[i-1],
                                        hidden_channels[i], masks[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)

    def forward(self, x, training):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
