# %% imports
from dataclasses import dataclass
from time import sleep, time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.backends.cudnn.benchmark = True

# Run IPython magic commands
from IPython.core.getipython import get_ipython
from torch.utils.data import DataLoader
from torchviz import make_dot
from tqdm.auto import tqdm, trange

import my_loaders
from mnist_networks import Autoencoder, AutoencoderModule, VariationalAutoencoder

torch.backends.cudnn.benchmark = True

# Run IPython magic commands
from IPython.core.getipython import get_ipython

ipython = get_ipython()
if ipython is not None:
    # Only works in interactive mode
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


@dataclass
class TrainingDiagnostics:
    num_total_steps: int
    tracked_loss: torch.Tensor
    tracked_lr: torch.Tensor


def trainAutoencoder(
    model: nn.Module,
    data: DataLoader,
    num_epochs: int = 4,
    learning_rate: float = 0.01,
    learning_rate_T_0: int = 10,
    clip_grad_max_norm: float = 1e9,
    loss_fn: Callable = nn.MSELoss(),
) -> tuple[nn.Module, TrainingDiagnostics]:

    optimiser = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=learning_rate_T_0
    )

    num_iter = len(data)
    num_total_steps = num_iter * num_epochs

    # Make only 1 bar for the inner iteration, so we can reuse it and not make new ones
    train_bar = tqdm(range(num_iter), desc="Iteration", position=1, leave=False)

    tracked_loss = torch.zeros((num_total_steps), requires_grad=False)
    tracked_lr = torch.zeros((num_total_steps), requires_grad=False)
    model.train()
    for epoch in trange(num_epochs, desc="Epoch", position=0):
        for idx, (inputs, targets) in enumerate(data):
            train_bar.update(1)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)

            # Optimise in batches (do 1 optim step per [batch_size] images)
            loss.backward()
            # Clip gradient to stop exploding
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), clip_grad_max_norm
            )
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            tracked_loss[epoch * num_iter + idx] = loss.detach()
            tracked_lr[epoch * num_iter + idx] = optimiser.param_groups[0]["lr"]

        train_bar.reset()
        scheduler.step()

    training_diagnostics = TrainingDiagnostics(
        num_total_steps=num_total_steps,
        tracked_loss=tracked_loss,
        tracked_lr=tracked_lr,
    )

    return model, training_diagnostics


def avgAutoencoderLoss(
    model: nn.Module, data: DataLoader, loss_fn: Callable = nn.MSELoss()
):
    model.eval()
    # Use a MSE loss when not training, even for variational autoencoder
    num_iter = len(data)
    tracked_loss = torch.zeros((num_iter), requires_grad=False)
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(data):
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            tracked_loss[idx] = loss

    avg_loss = torch.mean(tracked_loss)

    return avg_loss.cpu().item()


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


def plotLatentSpace(model, data: DataLoader, num_batches: int = 100):
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
    model,
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
    plt.imshow(img, extent=[*r0, *r1], cmap="gray")


def renderModelGraph(
    model: nn.Module,
    data: DataLoader,
    filepath: str = "network_graph",
    format: str = "pdf",
):
    model.eval()
    x, y = iter(data).__next__()
    test_batch = model(x.to(device))
    dot = make_dot(test_batch, params=dict(list(model.named_parameters())))
    dot.render(filepath, format=format, cleanup=True)


# <<< Definitions <<<

# Load MNIST training data
train_loader, test_loader = my_loaders.mnist(batch_size=128)
# Define and train model
# model = Autoencoder(latent_dims=2).to(device)
model = VariationalAutoencoder(latent_dims=2, dropout_prob=0).to(device)
model, training_diagnostics = trainAutoencoder(
    model,
    data=train_loader,
    num_epochs=2,
    learning_rate=1e-3,
    clip_grad_max_norm=1e-4,
    loss_fn=model.loss_fn,
)

# % Diagnostics
fig, ax = plt.subplots(2, 1)
plt.subplot(2, 1, 1)
plt.plot(range(training_diagnostics.num_total_steps), training_diagnostics.tracked_loss)
plt.xlabel("step")
plt.ylabel("loss")
plt.ylim((0, 10000))
plt.subplot(2, 1, 2)
plt.plot(range(training_diagnostics.num_total_steps), training_diagnostics.tracked_lr)
plt.xlabel("step")
plt.ylabel("learning rate")

# %
compareImages(model, test_loader)
plotLatentSpace(model, test_loader)
plotReconstructed(model, (-2, 2), (-2, 2), 20)
renderModelGraph(model, test_loader)

plt.show()

print(
    f"Final train loss: {avgAutoencoderLoss(model, train_loader, loss_fn = nn.MSELoss()) :.3g}"
)
print(
    f"Final test loss: {avgAutoencoderLoss(model, test_loader, loss_fn = nn.MSELoss()) :.3g}"
)
