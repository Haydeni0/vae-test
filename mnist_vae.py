# %% imports
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# Run IPython magic commands
from IPython.core.getipython import get_ipython
from torch.utils.data import DataLoader
from torchviz import make_dot
from tqdm.auto import tqdm, trange

import my_loaders

ipython = get_ipython()
if ipython is not None:
    # Only works in interactive mode
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# >>> Definitions >>>
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, latent_dims)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 28 * 28)

    def forward(self, x: torch.Tensor):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
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


def train(
    model: Autoencoder,
    data: DataLoader,
    num_epochs: int = 4,
    learning_rate: float = 0.01,
):

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    num_iter = len(data)
    num_total_steps = num_iter * num_epochs

    # Make only 1 bar for the inner iteration, so we can reuse it and not make new ones
    train_bar = tqdm(range(num_iter), desc="Iteration", position=1, leave=False)

    tracked_loss = torch.zeros((num_total_steps), requires_grad=False)
    model.train()
    for epoch in trange(num_epochs, desc="Epoch", position=0):
        for idx, (images, labels) in enumerate(data):
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


def compareImages(model: nn.Module, data: DataLoader, num_images: int = 10):
    model.eval()
    # Plot imput images compared to reconstructed images
    image = iter(data).__next__()[0][:num_images].reshape(-1, 28, 28)
    with torch.no_grad():
        predicted_image = model(image.to(device))
    fig, ax = plt.subplots(2, num_images)
    for img_idx in range(num_images):
        plt.subplot(2, num_images, img_idx + 1)
        plt.imshow(image[img_idx], cmap="gray")
        plt.subplot(2, num_images, num_images + img_idx + 1)
        plt.imshow(predicted_image[img_idx].cpu().reshape(28, 28), cmap="gray")

    return fig, ax


def plotLatentSpace(model: Autoencoder, data: DataLoader, num_batches: int = 100):
    model.eval()
    fig, ax = plt.subplots()
    with torch.no_grad():
        for idx, (x, y) in enumerate(data):
            z = model.encoder(x.to(device))
            z = z.cpu()

            plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")

            if idx > num_batches:
                break

    plt.colorbar()


def plotReconstructed(
    model: Autoencoder,
    r0: tuple[float, float] = (-5, 10),
    r1: tuple[float, float] = (-10, 5),
    n: int = 12,
):
    fig, ax = plt.subplots()
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = model.decoder(z)
            x_hat = x_hat.reshape(28, 28).to("cpu").detach().numpy()
            img[(n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1], cmap = "gray")

def renderModelGraph(model: nn.Module, data: DataLoader, filepath: str = "network_graph", format: str = "pdf"):
    model.eval()
    x, y = iter(data).__next__()
    test_batch = model(x.to(device))
    dot = make_dot(test_batch, params=dict(list(model.named_parameters())))
    dot.render("network_graph", format="pdf", cleanup=True)

# <<< Definitions <<<

# Load MNIST training data
train_loader, test_loader = my_loaders.mnist(batch_size=100)
# Define and train model
model = Autoencoder(latent_dims=2).to(device)
model, tracked_loss = train(model, data=train_loader, num_epochs=40, learning_rate=1e-4)

# % Diagnostics
fig, ax = plt.subplots()
plt.plot(range(len(tracked_loss)), tracked_loss)

compareImages(model, test_loader)
plotLatentSpace(model, test_loader)
plotReconstructed(model, (-5, 5), (-5, 5), 20)
renderModelGraph(model, test_loader)

plt.show()

print(f"Final loss: {tracked_loss[-1] :.3g}")
