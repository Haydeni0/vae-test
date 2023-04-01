# %% imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm, trange
from time import time, sleep
import multiprocessing as mp

import matplotlib.pyplot as plt

# Run IPython magic commands
from IPython.core.getipython import get_ipython

ipython = get_ipython()
if ipython is not None:
    # Only works in interactive mode
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# %% Load MNIST training data

# Load and transform to tensor
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

# Loaders
BATCH_SIZE = 100
# The loading can be a large bottleneck for the training speed. Increase number of workers and pin memory.
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=mp.cpu_count(),
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=mp.cpu_count(),
    pin_memory=True,
)

# Plot
loaded_test_data = iter(test_loader)
example_data, example_targets = loaded_test_data.__next__()
fig, ax = plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap="gray")


# %% Define model
image_size = 28 * 28
num_classes = 10
learning_rate = 0.01


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.FC1 = nn.Linear(image_size, 500)
        self.act1 = nn.ReLU()
        self.FC2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.FC1(x)
        x = self.act1(x)
        x = self.FC2(x)
        # No activation function due to the use of Adam optimiser or loss function?
        return x


model = MnistNet().to(device)

loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# %% Train
num_epochs = 100

model.train()

num_total_steps = len(train_loader)
for epoch in trange(num_epochs, desc="Epoch"):
    for idx, (images, labels) in tqdm(enumerate(train_loader), desc="Iteration"):
        images = images.reshape(-1, image_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)

        # Optimise in batches (do 1 optim step per [batch_size] images)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()


# %% Test
model.eval()

# Don't compute gradients
with torch.no_grad():
    num_correct = 0

    for images, labels in test_loader:
        images = images.reshape(-1, image_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, pred_idx = torch.max(outputs, 1)
        num_correct += (pred_idx == labels).sum().item()
    
    accuracy = num_correct / len(test_loader)
    print(f"Accuracy: {accuracy}%")

        

# %% tqdm test

h = trange(10, desc="Epoch")
m = trange(5, desc="Iteration")

for _ in h:
    for __ in m:
        sleep(0.2)
