# %% imports
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
train_loader, test_loader = my_loaders.mnist(BATCH_SIZE)

# Autoencoder

image_size = 28 * 28
num_classes = 10

class MnistAe(nn.Module):
    def __init__(self):
        super(MnistAe, self).__init__()

        self.fc1 = nn.Linear(image_size, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, 20)
        self.fc4 = nn.Linear(20, 80)
        self.fc5 = nn.Linear(80, 200)
        self.fc6 = nn.Linear(200, image_size)
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        return x

    def imgReshape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, image_size)
        return x

model = MnistAe().to(device)

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(params=model.parameters(), lr=0.01)


num_epochs = 4
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
        images = model.imgReshape(images)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, images)

        # Optimise in batches (do 1 optim step per [batch_size] images)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

        tracked_loss[epoch * num_iter + idx] = loss.detach()
    train_bar.reset()

# Training diagnostics

fig, ax = plt.subplots()
plt.plot(range(len(tracked_loss)), tracked_loss)


# %% Test

model.eval()

num_images = 10
# Plot imput images compared to reconstructed images
image = iter(test_loader).__next__()[0][:num_images].reshape(-1, 28, 28)
predicted_image = model(model.imgReshape(image.to(device)))
fig, ax = plt.subplots(2, num_images)
for img_idx in range(num_images):
    plt.subplot(2, num_images, img_idx + 1)
    plt.imshow(image[img_idx], cmap="gray")
    plt.subplot(2, num_images, num_images + img_idx + 1)
    plt.imshow(predicted_image[img_idx].detach().cpu().reshape(28,28), cmap="gray")

# %% Visualise model
from torchviz import make_dot

dot = make_dot(predicted_image, params=dict(list(model.named_parameters())))
dot.render("network_graph", format="pdf", cleanup=True)