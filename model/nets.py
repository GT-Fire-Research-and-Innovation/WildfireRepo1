"""Encoder-decoder network shared by the plain U-Net baseline and the hybrid model."""
import torch
import torch.nn as nn


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())


class UNet(nn.Module):
    """4-level U-Net. Input sides must be multiples of 8."""

    def __init__(self, c_in, c_out, width=32):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.enc = nn.ModuleList([block(c_in, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i], w[i - 1], 2, stride=2) for i in (3, 2, 1)])
        self.dec = nn.ModuleList([block(w[i - 1] * 2, w[i - 1]) for i in (3, 2, 1)])
        self.head = nn.Conv2d(w[0], c_out, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            skips.append(x)
            if i < 3:
                x = self.pool(x)
        for j, i in enumerate((2, 1, 0)):
            x = self.dec[j](torch.cat([self.up[j](x), skips[i]], 1))
        return self.head(x)
