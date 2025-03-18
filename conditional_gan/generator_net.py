import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear_1 = nn.Linear(100, 7*7*128)
        self.embedding_1 = nn.Linear(1, 50)
        self.linear_2 = nn.Linear(50, 49)

        self.deconv_1 = nn.ConvTranspose2d(129, 64, (2, 2), 2)
        self.deconv_2 = nn.ConvTranspose2d(64, 32, (2, 2), 2)
        self.deconv_3 = nn.Conv2d(32, 1, (1, 1))

        self.relu = nn.LeakyReLU(0.2)

        self.drop = nn.Dropout(0.5)

    def forward(self, noise, label):
        x = self.linear_1(noise)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 128, 7, 7))
        x = self.drop(x)

        y = self.embedding_1(label)
        y = self.relu(y)
        y = self.drop(y)
        y = self.linear_2(y)
        y = torch.reshape(y, (-1, 1, 7, 7))
        y = self.relu(y)
        y = self.drop(y)

        z = torch.concat((x, y), dim=1)

        z = self.deconv_1(z)
        z = self.deconv_2(z)
        z = self.deconv_3(z)

        return z

