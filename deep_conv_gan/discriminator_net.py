import torch.nn as nn
import torch.nn.functional as F
import torch

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.fc = nn.Linear(512, 1)

        self.conv_1 = nn.Conv2d(3, 512, kernel_size=3)
        self.conv_2 = nn.Conv2d(512, 256, kernel_size=3)
        self.conv_3 = nn.Conv2d(256, 128, kernel_size=3)

        self.max_pool = nn.MaxPool2d((2, 2))

        self.Lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool(x)
        x = self.Lrelu(x)

        x = self.conv_2(x)
        x = self.max_pool(x)
        x = self.Lrelu(x)

        x = self.conv_3(x)
        x = self.max_pool(x)
        x = self.Lrelu(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        x = self.sigmoid(x)

        return x
    

# D = DiscriminatorNet()

# noise = torch.randn(4, 3, 32, 32)

# print(D(noise).shape)



