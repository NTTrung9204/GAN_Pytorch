import torch.nn as nn
import torch.nn.functional as F
import torch

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        self.fc = nn.Linear(128, 2048)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))

        self.deconv_1 = nn.ConvTranspose2d(512, 256, (3, 3), 2, 1)
        self.deconv_2 = nn.ConvTranspose2d(256, 128, (3, 3), 2)
        self.deconv_3 = nn.ConvTranspose2d(128, 64, (3, 3), 2)
        self.deconv_4 = nn.ConvTranspose2d(64, 3, (3, 3), 2)

        self.batchNorm_1 = nn.BatchNorm2d(512)
        self.batchNorm_2 = nn.BatchNorm2d(256)
        self.batchNorm_3 = nn.BatchNorm2d(128)
        self.batchNorm_4 = nn.BatchNorm2d(64)
        self.batchNorm_5 = nn.BatchNorm2d(3)

        self.Lrelu = nn.LeakyReLU(0.2)

        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)

        x = self.fc(x) # 128 -> 2048
        x = x.view(batch_size, 512, 2, 2) # 2048 -> (2, 2, 512)
        x = self.batchNorm_1(x)
        x = self.Lrelu(x)

        x = self.deconv_1(x) # (4, 4, 256)
        x = self.batchNorm_2(x)
        x = self.Lrelu(x)

        x = self.deconv_2(x) # (8, 8, 128)
        x = self.batchNorm_3(x)
        x = self.Lrelu(x)

        x = self.deconv_3(x) # (16, 16, 64)
        x = self.batchNorm_4(x)
        x = self.Lrelu(x)

        x = self.deconv_4(x) # (32, 32, 3)
        x = self.pad(x)
        x = self.batchNorm_5(x)

        x = self.tanh(x)

        return x


# G = GeneratorNet()

# noise = torch.randn(4, 128)

# print(G(noise).shape)