import torch.nn as nn


class Unflatten(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()

        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(-1, self.C, self.H, self.W)
