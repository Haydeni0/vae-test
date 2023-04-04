
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# >>> Definitions >>>

# Largely based off https://avandekleut.github.io/vae/


class Encoder(nn.Module):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, latent_dims)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = F.gelu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4_mu = nn.Linear(256, latent_dims)
        self.fc4_sigma = nn.Linear(256, latent_dims)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.N = torch.distributions.Normal(0, 1)
        
        self.kl = 0

    def forward(self, x: torch.Tensor):
        if x.get_device() >= 0:
            # Hack to get sampling on GPU
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = F.gelu(self.fc3(x))
        x = self.dropout(x)

        mu = self.fc4_mu(x)
        sigma = torch.exp(self.fc4_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()

        return z


class ConvolutionalVariationalEncoder(nn.Module):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(ConvolutionalVariationalEncoder, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 64, 7, padding="same")
        self.conv2 = nn.Conv2d(64, 32, 7, padding="same")
        self.fc1 = nn.Linear(32*7*7, 256)
        self.fc_mu = nn.Linear(256, latent_dims)
        self.fc_sigma = nn.Linear(256, latent_dims)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.N = torch.distributions.Normal(0, 1)
        
        self.kl = 0

    def forward(self, x: torch.Tensor):
        if x.get_device() >= 0:
            # Hack to get sampling on GPU
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

        # [N, 1, 28, 28]
        x = self.dropout(x)
        x = F.gelu(self.conv1(x)) # [N, 32, 28, 28]
        x = self.pool(x) # [N, 32, 14, 14]

        x = self.dropout(x)
        x = F.gelu(self.conv2(x)) # [N, 32, 14, 14]
        x = self.pool(x) # [N, 32, 7, 7]
        x = torch.flatten(x, 1) # [N, 32*7*7]

        x = self.dropout(x) # [N, 256]
        x = F.gelu(self.fc1(x))

        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()

        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = F.gelu(self.fc3(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc4(x))
        x = x.reshape(-1, 1, 28, 28)

        return x

class ConvolutionalDecoder(nn.Module):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(ConvolutionalDecoder, self).__init__()

        self.fc1 = nn.Linear(latent_dims, 256)
        self.fc2 = nn.Linear(256, 32*7*7)
        self.conv1 = nn.Conv2d(32, 64, 7, padding="same")
        self.conv2 = nn.Conv2d(64, 32, 7, padding="same")
        self.conv3 = nn.Conv2d(32, 1, 7, padding="same")

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor):
        # [N, latent_dims]
        x = self.dropout(x)
        x = F.gelu(self.fc1(x)) # [N, 256]

        x = self.dropout(x)
        x = F.gelu(self.fc2(x)) # [N, 32*7*7]
        x = x.reshape(-1, 32, 7, 7) # [N, 32, 7, 7]

        x = self.dropout(x)
        x = F.gelu(self.conv1(x)) # [N, 64, 7, 7]
        x = self.upsample1(x) # [N, 64, 14, 14]
        x = self.dropout(x)
        x = F.gelu(self.conv2(x)) # [N, 32, 14, 14]
        x = self.upsample2(x) # [N, 32, 28, 28]

        x = self.dropout(x)
        x = F.gelu(self.conv3(x)) # [N, 1, 28, 28]

        return x



class AutoencoderModule(nn.Module):
    """Base class for autoencoder and variational autoencoder"""

    encoder: Encoder | VariationalEncoder | ConvolutionalVariationalEncoder
    decoder: Decoder | ConvolutionalDecoder

    def __init__(self):
        super(AutoencoderModule, self).__init__()

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Autoencoder(AutoencoderModule):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(
            Autoencoder,
            self,
        ).__init__()

        self.encoder = Encoder(latent_dims, dropout_prob=dropout_prob)
        self.decoder = Decoder(latent_dims, dropout_prob=dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor):
        return ((x - y) ** 2).sum()


class VariationalAutoencoder(AutoencoderModule):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = VariationalEncoder(latent_dims, dropout_prob=dropout_prob)
        self.decoder = Decoder(latent_dims, dropout_prob=dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor):
        return ((x - y) ** 2).sum() + self.encoder.kl
    
class ConvolutionalVariationalAutoencoder(AutoencoderModule):
    def __init__(self, latent_dims: int, dropout_prob: float = 0):
        super(ConvolutionalVariationalAutoencoder, self).__init__()

        self.encoder = ConvolutionalVariationalEncoder(latent_dims, dropout_prob=dropout_prob)
        # Seems to have less blurry images with regular decoder
        self.decoder = Decoder(latent_dims, dropout_prob=dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor):
        return ((x - y) ** 2).sum() + self.encoder.kl