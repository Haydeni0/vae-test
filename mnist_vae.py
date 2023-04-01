# %% imports
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

import my_loaders

# Run IPython magic commands
from IPython.core.getipython import get_ipython
from tqdm.auto import tqdm, trange

ipython = get_ipython()
if ipython is not None:
    # Only works in interactive mode
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Load MNIST training data
# Loaders
BATCH_SIZE = 100
train_loader, test_loader = my_loaders.mnist(batch_size=BATCH_SIZE)


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, latent_dims)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 512)
        self.fc2 = nn.Linear(512, 28 * 28)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = x.reshape(-1, 1, 28, 28)

        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def train(model: nn.Module, train_loader: DataLoader, num_epochs: int = 4):

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(params=model.parameters(), lr=0.01)

    num_iter = len(train_loader)
    num_total_steps = num_iter * num_epochs

    # Make only 1 bar for the inner iteration, so we can reuse it and not make new ones
    train_bar = tqdm(range(num_iter), desc="Iteration", position=1, leave=False)

    tracked_loss = torch.zeros((num_total_steps), requires_grad=False)
    model.train()
    for epoch in trange(num_epochs, desc="Epoch", position=0):
        for idx, (images, labels) in enumerate(train_loader):
            train_bar.update(1)
            images = images.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, images)

            # Optimise in batches (do 1 optim step per [batch_size] images)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            tracked_loss[epoch * num_iter + idx] = loss.detach()
        train_bar.reset()
    
    return model, tracked_loss


model = Autoencoder(latent_dims=10).to(device)
model, tracked_loss = train(model, train_loader=train_loader, num_epochs=4)

# Training diagnostics

fig, ax = plt.subplots()
plt.plot(range(len(tracked_loss)), tracked_loss)
print(f"Loss: {tracked_loss[-1] :.3g}")

#  Test

model.eval()

num_images = 10
# Plot imput images compared to reconstructed images
image = iter(test_loader).__next__()[0][:num_images].reshape(-1, 28, 28)
with torch.no_grad():
    predicted_image = model(image.to(device))
fig, ax = plt.subplots(2, num_images)
for img_idx in range(num_images):
    plt.subplot(2, num_images, img_idx + 1)
    plt.imshow(image[img_idx], cmap="gray")
    plt.subplot(2, num_images, num_images + img_idx + 1)
    plt.imshow(predicted_image[img_idx].cpu().reshape(28, 28), cmap="gray")