import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
from discriminator_net import DiscriminatorNet
from generator_net import GeneratorNet
import torchvision

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is used: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    batch_size = 256

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)

    print("Train size:", len(train_dataset))

    G_Net = GeneratorNet().to(device)
    D_Net = DiscriminatorNet().to(device)

    G_criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G_Net.parameters(), lr=0.0002)

    D_criterion = nn.BCELoss()
    D_optimizer = optim.Adam(D_Net.parameters(), lr=0.0002)

    epochs = 100
    z_dim = 100
    G_losses = [10]
    D_losses = []
    for epoch in range(epochs):
        for real_images, _ in train_loader:
            real_labels = torch.ones(batch_size, 1).to(device)
            real_images = real_images.flatten(1).to(device)

            noise = torch.randn(batch_size, z_dim).to(device)

            fake_images = G_Net(noise)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)

            train_images_D = torch.cat((real_images, fake_images))
            train_labels_D = torch.cat((real_labels, fake_labels))

            indices = torch.randperm(train_images_D.size(0))
            train_images_D = train_images_D[indices]
            train_labels_D = train_labels_D[indices]

            D_Net.train()
            D_prediction = D_Net(train_images_D)

            D_loss = D_criterion(D_prediction, train_labels_D)

            D_losses.append(D_loss.item())

            D_Net.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            if D_loss.item() < 1:
                D_Net.eval()
                noise = torch.randn(batch_size, z_dim).to(device)
                G_labels = torch.ones(batch_size, 1).to(device) - 0.1

                G_outputs = G_Net(noise)
                D_prediction = D_Net(G_outputs)

                G_loss = G_criterion(D_prediction, G_labels)

                G_losses.append(G_loss.item())

                G_Net.zero_grad()
                G_loss.backward()
                G_optimizer.step()

            sys.stdout.write(f"\rG loss: {G_losses[-1]:.5f} | D loss: {D_losses[-1]:.5f}")

        print()

        if epoch % 10 == 0:
            noise = torch.rand(25, z_dim).to(device)
            with torch.no_grad():
                fake_images = G_Net(noise).view(-1, 1, 28, 28)

            grid = torchvision.utils.make_grid(fake_images, nrow=5, normalize=True)

            plt.figure(figsize=(6, 6))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.title(f"Generated Images at Epoch {epoch+1}")
            plt.show()



    plt.plot(G_losses)
    plt.plot(D_losses)

    plt.xlabel("Epochs")
    plt.ylabel("Losses")

    plt.title("GAN Net Training")

    plt.show()

    torch.save(G_Net.state_dict(), "trained_model/G_Net_v7.pth")
    torch.save(D_Net.state_dict(), "trained_model/D_Net_v7.pth")
