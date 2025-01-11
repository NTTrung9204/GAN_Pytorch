import torch.nn as nn
import torch.nn.functional as F
import torch

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.fcs_1 = nn.Linear(28**2, 1024)
        self.drop_1 = nn.Dropout(0.5)
        self.fcs_2 = nn.Linear(1024, 512)
        self.drop_2 = nn.Dropout(0.3)
        self.fcs_3 = nn.Linear(512, 256)
        self.drop_3 = nn.Dropout(0.1)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fcs_1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fcs_2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fcs_3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.output(x)
        
        return torch.sigmoid(x)
