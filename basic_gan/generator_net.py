import torch.nn as nn
import torch.nn.functional as F
import torch

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        self.fcs_1 = nn.Linear(100, 256)
        self.drop = nn.Dropout(0.3)
        self.fcs_2 = nn.Linear(256, 512)
        self.fcs_3 = nn.Linear(512, 1024)
        self.output = nn.Linear(1024, 28**2)
        self.batchnorm_1 = nn.BatchNorm1d(256)
        self.batchnorm_2 = nn.BatchNorm1d(512)
        self.batchnorm_3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.fcs_1(x)
        # x = self.batchnorm_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.drop(x)

        x = self.fcs_2(x)
        # x = self.batchnorm_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.drop(x)

        x = self.fcs_3(x)
        # x = self.batchnorm_3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.drop(x)

        x = self.output(x)

        return torch.sigmoid(x)
