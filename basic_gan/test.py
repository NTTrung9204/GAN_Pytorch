import torch
import torchvision
from discriminator_net import DiscriminatorNet
from generator_net import GeneratorNet
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_Net = GeneratorNet().to(device)
G_Net.load_state_dict(torch.load("trained_model/G_Net_v2.pth"))

z_dim = 100

G_Net.eval()

noise = torch.rand(1, z_dim).to(device)

fake_images = G_Net(noise).view(-1, 28, 28).detach()

plt.figure(figsize=(8, 8))
plt.imshow(fake_images.permute(1, 2, 0).to("cpu"))
plt.axis('off')
plt.title(f"Generated Images")
plt.show()
