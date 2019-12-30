import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_size, num_channels, num_additional_features=0):
        super().__init__()

        num_convolutions, encoded_size = 0, image_size
        while num_convolutions < 3 and encoded_size % 2 == 0:
            encoded_size //= 2
            num_convolutions += 1
        assert num_convolutions >= 2

        self._num_additional_features = num_additional_features

        convolutions, in_channels, out_channels = [], num_channels, 32
        for i in range(num_convolutions):
            convolution = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                          stride=2, padding=1),
                nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.01)
            )
            convolutions.append(convolution)
            in_channels = out_channels
            out_channels = in_channels * 2
        self.convolutions = nn.Sequential(*convolutions)

        self.affine_1 = nn.Sequential(
            nn.Linear(in_channels * encoded_size * encoded_size + num_additional_features, 1024),
            nn.LeakyReLU(0.01)
        )
        self.affine_2 = nn.Linear(1024, 1)

    def forward(self, x, additional_features=None):
        num_additional_features = 0 if additional_features is None else additional_features.size(1)
        assert num_additional_features == self._num_additional_features

        out = self.convolutions(x)
        out = out.view(x.size(0), -1)

        if additional_features is not None:
            out = torch.cat([out, additional_features], dim=1)

        out = self.affine_1(out)
        out = self.affine_2(out)
        return out
