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


class WindowAttention(nn.Module):
    """Self-attention inside ws x ws windows with a learned bias per relative offset (Swin, Liu et al. 2021)."""

    def __init__(self, dim, heads, ws):
        super().__init__()
        self.heads, self.ws, self.scale = heads, ws, (dim // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.rel = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel, std=0.02)
        yy, xx = torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        c = torch.stack([yy.flatten(), xx.flatten()])
        d = c[:, :, None] - c[:, None, :] + ws - 1
        self.register_buffer("rel_index", d[0] * (2 * ws - 1) + d[1], persistent=False)

    def forward(self, x, mask):                               # x (windows, ws*ws, C); mask (windows, N, N) or None
        Bw, N, C = x.shape
        q, k, v = self.qkv(x).view(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = (q * self.scale) @ k.transpose(-2, -1) + self.rel[self.rel_index].permute(2, 0, 1)[None].to(x.dtype)
        if mask is not None:
            a = a + mask[:, None].to(a.dtype)
        return self.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(Bw, N, C))


class SwinBlock(nn.Module):
    """One Swin block on a (B, H, W, C) map of any size. The map is padded to whole windows; with shift > 0
    the windows are offset by half a window so information crosses the previous block's window borders.
    Padding, and cells that only became neighbours through the cyclic shift, are masked out."""

    def __init__(self, dim, heads, ws=8, shift=0, mlp=2, dropout=0.1):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, ws)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mlp, dim))
        self.drop = nn.Dropout(dropout)
        self._masks = {}

    def windows(self, x):                                      # (B, Hp, Wp, C) -> (B * windows, ws*ws, C)
        B, Hp, Wp, C = x.shape
        ws = self.ws
        return x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)

    def mask(self, H, W, Hp, Wp, device):
        key = (H, W, str(device))
        if key not in self._masks:
            ws, s = self.ws, self.shift
            region = torch.zeros(Hp, Wp, device=device)
            region[H:, :] = 100                                # padding
            region[:, W:] = 100
            if s:
                region = torch.roll(region, (-s, -s), (0, 1))
                k = 0
                for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                    for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                        region[hs, wsl] += k                   # the wrapped-around strips are their own regions
                        k += 1
            r = self.windows(region[None, :, :, None]).squeeze(-1)             # (windows, N)
            m = (r[:, :, None] != r[:, None, :]).float() * -100.0
            self._masks[key] = m if (s or Hp != H or Wp != W) else None
            if len(self._masks) > 64:
                self._masks.pop(next(iter(self._masks)))
        return self._masks[key]

    def forward(self, x):
        B, H, W, C = x.shape
        ws, s = self.ws, self.shift
        pb, pr = (-H) % ws, (-W) % ws
        y = self.n1(x)
        if pb or pr:
            y = nn.functional.pad(y, (0, 0, 0, pr, 0, pb))
        Hp, Wp = H + pb, W + pr
        if s:
            y = torch.roll(y, (-s, -s), (1, 2))
        m = self.mask(H, W, Hp, Wp, x.device)
        a = self.attn(self.windows(y), None if m is None else m.repeat(B, 1, 1))
        a = a.view(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if s:
            a = torch.roll(a, (s, s), (1, 2))
        x = x + self.drop(a[:, :H, :W])
        return x + self.drop(self.mlp(self.n2(x)))


class SwinStage(nn.Module):
    def __init__(self, dim, depth, heads, ws=8, mlp=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, ws, 0 if k % 2 == 0 else ws // 2, mlp, dropout) for k in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                                      # (B, C, H, W) -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        for b in self.blocks:
            x = b(x)
        return self.norm(x).permute(0, 3, 1, 2)


class SwinUNet(nn.Module):
    """Swin transformer encoder inside a U-Net (the Swin-UNETR layout). The two finest levels stay
    convolutional, so the 150 m edge detail is kept; the two coarser levels (600 m and 1.2 km cells) are
    Swin stages, where each cell looks across an 8 x 8 window and shifted windows pass information on,
    block after block. Unlike the transformer bottleneck of UNet3, which attends over the WHOLE map (and so
    behaves differently on a 24 km training crop and a 115 km evaluation window), window attention sees
    the same neighbourhood at any map size, and its cost grows with area, not area squared.
    Same inputs and history handling as UNet3 (the day-by-day history is the last c_seq channels).
    Input sides must be multiples of 8."""

    def __init__(self, c_in, c_seq=0, days=6, width=64, depths=(2, 4), heads=(8, 16), ws=8, dropout=0.1, d_time=32):
        super().__init__()
        self.c_seq, self.days = c_seq, days
        self.c_static = c_in - c_seq
        self.temporal = TemporalAttention(c_seq // days, days, d_time) if c_seq else None
        c0 = self.c_static + (d_time if c_seq else 0)
        w = [width, width * 2, width * 4, width * 8]
        self.enc = nn.ModuleList([block(c0, w[0]), block(w[0], w[1])])
        self.pool = nn.MaxPool2d(2)
        self.merge = nn.ModuleList([nn.Sequential(nn.Conv2d(w[1], w[2], 2, stride=2), nn.GroupNorm(8, w[2])),
                                    nn.Sequential(nn.Conv2d(w[2], w[3], 2, stride=2), nn.GroupNorm(8, w[3]))])
        self.swin = nn.ModuleList([SwinStage(w[2], depths[0], heads[0], ws, dropout=dropout),
                                   SwinStage(w[3], depths[1], heads[1], ws, dropout=dropout)])
        self.drop = nn.Dropout2d(dropout)
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i], w[i - 1], 2, stride=2) for i in (3, 2, 1)])
        self.dec = nn.ModuleList([block(w[i - 1] * 2, w[i - 1]) for i in (3, 2, 1)])
        self.head = nn.Conv2d(w[0], 1, 1)
        for m in self.swin.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        static = x[:, :self.c_static]
        if self.temporal is not None:
            B, _, H, W = x.shape
            seq = x[:, self.c_static:].reshape(B, self.days, self.c_seq // self.days, H, W)
            x = torch.cat([static, self.temporal(seq)], 1)
        else:
            x = static
        s0 = self.enc[0](x)                                    # 150 m
        s1 = self.enc[1](self.pool(s0))                        # 300 m
        s2 = self.swin[0](self.merge[0](s1))                   # 600 m, window = 4.8 km
        s3 = self.drop(self.swin[1](self.merge[1](s2)))        # 1.2 km, window = 9.6 km
        x = s3
        for j, skip in enumerate((s2, s1, s0)):
            x = self.dec[j](torch.cat([self.up[j](x), skip], 1))
        return self.head(x)
