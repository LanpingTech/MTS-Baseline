import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, timesteps, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 6, 5, padding='same')
        self.pool1 = torch.nn.AdaptiveAvgPool1d(timesteps)
        self.conv2 = torch.nn.Conv1d(6, 16, 5, padding='same')
        self.pool2 = torch.nn.AdaptiveAvgPool1d(timesteps)
        self.linear1 = torch.nn.Linear(timesteps * 16, 128)
        self.linear2 = torch.nn.Linear(128, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out=F.normalize(out)
        out = F.relu(out)
        out = self.linear2(out)
        out=F.normalize(out)
        out = F.relu(out)
        return out  #判断一下这里输出shape

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, timestep, out_channels):
        super(Decoder, self).__init__()
        self.timestep = timestep
        self.conv1 = torch.nn.Conv1d(16, 16, 5, padding='same')
        self.conv2 = torch.nn.Conv1d(16, in_channels, 5, padding='same')
        self.linear = torch.nn.Linear(out_channels, timestep * 16)


    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), -1, self.timestep)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        return out #判断一下这里输出shape
