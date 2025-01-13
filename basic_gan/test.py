import torch
import torchvision
from discriminator_net import DiscriminatorNet
from generator_net import GeneratorNet
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_Net = GeneratorNet().to(device)
# G_Net.load_state_dict(torch.load("trained_model/G_Net_v2.pth"))

z_dim = 100

G_Net.eval()

# noise = torch.rand(1, z_dim).to(device)

# fake_images = G_Net(noise).view(-1, 28, 28).detach()

# plt.figure(figsize=(4, 4))
# plt.imshow(fake_images.permute(1, 2, 0).to("cpu"), cmap='gray')
# plt.axis('off')
# plt.title(f"Generated Images")
# plt.show()

noise = torch.rand(16, z_dim).to(device)
with torch.no_grad():
    fake_images = G_Net(noise).view(-1, 1, 28, 28)

grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)

plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis('off')
plt.title(f"Generated Images")
plt.show()