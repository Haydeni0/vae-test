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

# %% Load MNIST training data
# Loaders
BATCH_SIZE = 100

train_loader, test_loader = my_loaders.mnist(BATCH_SIZE)

# Plot
loaded_test_data = iter(test_loader)
example_data, example_targets = loaded_test_data.__next__()
fig, ax = plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap="gray")


# %% Define and train model
image_size = 28 * 28
num_classes = 10
learning_rate = 0.01


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.fc1 = nn.Linear(image_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, image_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # No activation function due to the use of cross entropy loss
        return x


class MnistConvNet(nn.Module):
    def __init__(self):
        super(MnistConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        self.fc1 = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor):
        # Input: [N, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [N, 32, 28, 28]
        x = self.pool(x)  # [N, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [N, 64, 14, 14]
        x = self.pool(x)  # [N, 64, 7, 7]
        x = x.reshape(-1, 64 * 7 * 7) # [N, 64*7*7]
        x = self.fc1(x) # [N, num_classes]

        return x


model = MnistNet().to(device)
# model = MnistConvNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


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
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

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

print()
# Don't compute gradients
with torch.no_grad():
    num_correct = 0
    num_samples = len(test_loader.dataset)  # type: ignore

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, pred_idx = torch.max(outputs, 1)
        num_correct += (pred_idx == labels).sum().item()

    accuracy = num_correct / num_samples
    print(f"Accuracy: {accuracy*100}%")
