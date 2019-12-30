import numpy as np

import torch
import torch.nn as nn

from cgan.helpers import unnormalize
from cgan.layers import Unflatten


class Generator(nn.Module):
    def __init__(self, z_dim, image_size, num_channels, num_additional_features=0):
        super().__init__()

        num_deconvolutions, encoded_size, encoded_num_channels = 0, image_size, 16
        while num_deconvolutions < 3 and encoded_size % 2 == 0:
            encoded_size //= 2
            num_deconvolutions += 1
            encoded_num_channels *= 2
        assert num_deconvolutions >= 2

        self.z_dim = z_dim
        self._num_additional_features = num_additional_features

        self.affine_1 = nn.Sequential(
            nn.Linear(z_dim + num_additional_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.affine_2 = nn.Sequential(
            nn.Linear(1024, encoded_num_channels * encoded_size * encoded_size),
            nn.BatchNorm1d(encoded_num_channels * encoded_size * encoded_size),
            nn.ReLU()
        )
        self.unflatten = Unflatten(C=encoded_num_channels, H=encoded_size, W=encoded_size)

        deconvolutions, in_channels = [], encoded_num_channels
        for _ in range(num_deconvolutions - 1):
            out_channels = in_channels // 2
            deconvolution = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                   stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            deconvolutions.append(deconvolution)
            in_channels = out_channels
        deconvolutions.append(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_channels, kernel_size=4,
                               stride=2, padding=1)
        )
        self.deconvolutions = nn.Sequential(*deconvolutions)

    def forward(self, z, additional_features=None):
        num_additional_features = 0 if additional_features is None else additional_features.size(1)
        assert num_additional_features == self._num_additional_features

        if additional_features is not None:
            z = torch.cat([z, additional_features], dim=1)

        out = self.affine_1(z)
        out = self.affine_2(out)
        out = self.unflatten(out)

        out = self.deconvolutions(out)
        return torch.tanh(out)

    def generate(self, z, additional_features=None):
        images_tensor = self.forward(z=z, additional_features=additional_features)
        images_tensor = unnormalize(images_tensor)
        images = images_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
        images = (255 * images).round().astype(np.uint8)
        return images
