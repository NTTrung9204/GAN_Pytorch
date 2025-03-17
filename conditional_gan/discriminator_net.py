import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(1, 50)
        self.linear_1 = nn.Linear(50, 28*28)
        self.linear_2 = nn.Linear(7*7*128, 10)

        self.conv_1 = nn.Conv2d(2, 64, (2, 2), 2)
        self.conv_2 = nn.Conv2d(64, 128, (2, 2), 2)

    def forward(self, image, label):
        y = self.embedding(label)
        y = self.linear_1(y)

        y = torch.reshape(y, (1, 28, 28))

        z = torch.concat((image, y))

        z = self.conv_1(z)
        z = self.conv_2(z)

        z = torch.reshape(z, (1, 7*7*128))

        z = self.linear_2(z)

        return z
        