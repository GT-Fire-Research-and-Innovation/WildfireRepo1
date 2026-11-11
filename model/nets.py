"""Encoder-decoder network shared by the plain U-Net baseline and the hybrid model."""
import math

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


# ---------------------------------------------------------------------------- unet3
class TemporalAttention(nn.Module):
    """Day-by-day history -> one feature map. Each day's frame goes through the same small conv
    encoder; a learned day embedding is added; per-pixel attention over the days (several heads,
    as in L-TAE / U-TAE) picks which days matter at each location."""

    def __init__(self, c_frame, days, d=32, heads=4):
        super().__init__()
        self.enc = block(c_frame, d)
        self.day = nn.Parameter(torch.zeros(days, d))
        self.score = nn.Conv2d(d, heads, 1)
        self.d, self.heads = d, heads

    def forward(self, seq):                                  # (B, T, C, H, W)
        B, T, C, H, W = seq.shape
        z = self.enc(seq.flatten(0, 1)).view(B, T, self.d, H, W) + self.day[None, :T, :, None, None]
        a = torch.softmax(self.score(z.flatten(0, 1)).view(B, T, self.heads, 1, H, W), dim=1)
        z = z.view(B, T, self.heads, self.d // self.heads, H, W)
        return (a * z).sum(1).reshape(B, self.d, H, W)


def sincos_2d(h, w, d, device):
    """Fixed 2-D sine/cosine position codes (any grid size), (h*w, d)."""
    q = d // 4
    f = torch.exp(-math.log(1e4) * torch.arange(q, device=device) / q)
    yy, xx = torch.meshgrid(torch.arange(h, device=device).float(), torch.arange(w, device=device).float(), indexing="ij")
    ay, ax = yy.reshape(-1, 1) * f, xx.reshape(-1, 1) * f
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], 1)


class Bottleneck(nn.Module):
    """Transformer over the deepest feature map (TransUNet style): every location attends to every
    other one across the crop, plus a context token built from the crop-mean inputs (weather is
    near-uniform over a crop, so the token carries it). Residual, so it starts close to the U-Net."""

    def __init__(self, d, n_ctx, layers=4, heads=8, mlp=2, dropout=0.1):
        super().__init__()
        self.ctx = nn.Linear(n_ctx, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, d * mlp, dropout, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, ctx):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2) + sincos_2d(H, W, C, x.device)[None].to(x.dtype)
        t = torch.cat([self.ctx(ctx)[:, None], t], 1)
        for layer in self.layers:
            t = layer(t)
        t = self.norm(t[:, 1:])
        return x + t.transpose(1, 2).reshape(B, C, H, W)


class UNet3(nn.Module):
    """U-Net with (optionally) a day-by-day history read by temporal attention and a transformer
    bottleneck. The input is the usual stack with the history as its LAST c_seq channels
    (days x channels-per-day). Width 64 by default (~8M parameters without the extras)."""

    def __init__(self, c_in, c_seq=0, days=6, width=64, attn=True, layers=4, dropout=0.1, d_time=32):
        super().__init__()
        self.c_seq, self.days = c_seq, days
        self.c_static = c_in - c_seq
        self.temporal = TemporalAttention(c_seq // days, days, d_time) if c_seq else None
        c0 = self.c_static + (d_time if c_seq else 0)
        w = [width, width * 2, width * 4, width * 8]
        self.enc = nn.ModuleList([block(c0, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        self.bottleneck = Bottleneck(w[3], self.c_static, layers, dropout=dropout) if attn else None
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i], w[i - 1], 2, stride=2) for i in (3, 2, 1)])
        self.dec = nn.ModuleList([block(w[i - 1] * 2, w[i - 1]) for i in (3, 2, 1)])
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        static = x[:, :self.c_static]
        if self.temporal is not None:
            B, _, H, W = x.shape
            seq = x[:, self.c_static:].reshape(B, self.days, self.c_seq // self.days, H, W)
            x = torch.cat([static, self.temporal(seq)], 1)
        else:
            x = static
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            skips.append(x)
            if i < 3:
                x = self.pool(x)
        x = self.drop(x)
        if self.bottleneck is not None:
            x = self.bottleneck(x, static.mean((2, 3)))
        for j, i in enumerate((2, 1, 0)):
            x = self.dec[j](torch.cat([self.up[j](x), skips[i]], 1))
        return self.head(x)
