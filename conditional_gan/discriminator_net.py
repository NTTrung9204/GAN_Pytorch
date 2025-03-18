import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(1, 50)
        self.linear_1 = nn.Linear(50, 28*28)
        self.linear_2 = nn.Linear(7*7*128, 1)

        self.conv_1 = nn.Conv2d(2, 64, (2, 2), 2)
        self.conv_2 = nn.Conv2d(64, 128, (2, 2), 2)

        self.relu = nn.LeakyReLU(0.2)

        self.drop = nn.Dropout(0.5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, image, label):
        y = self.embedding(label)
        y = self.relu(y)
        y = self.linear_1(y)
        y = self.relu(y)

        y = torch.reshape(y, (-1, 1, 28, 28))

        z = torch.concat((image, y), dim=1)

        z = self.conv_1(z)
        z = self.conv_2(z)

        z = self.relu(z)
        z = torch.reshape(z, (-1, 1, 7*7*128))

        z = self.drop(z)

        z = self.linear_2(z)

        z = self.sigmoid(z)
        
        return z
        