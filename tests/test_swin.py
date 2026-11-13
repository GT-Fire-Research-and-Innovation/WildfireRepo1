"""Swin transformer network (model/nets.py SwinUNet)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import nets  # noqa: E402


def test_any_size_with_and_without_history():
    torch.manual_seed(0)
    for c_seq in (0, 24):
        m = nets.SwinUNet(40 + c_seq, c_seq, 6, width=16).eval()
        for h, w in ((160, 160), (168, 200), (64, 8)):          # multiples of 8, not of the window
            assert m(torch.randn(1, 40 + c_seq, h, w)).shape == (1, 1, h, w)


def test_shifted_windows_do_not_wrap_around_the_map():
    """The shift is done by rolling the map, which puts opposite edges side by side: the mask must keep
    them apart, or fire at the south-east corner would 'see' the north-west corner. (Tested on the Swin
    stage itself: the convolution blocks around it use GroupNorm, whose statistics span the whole map.)"""
    torch.manual_seed(0)
    st = nets.SwinStage(32, 4, 4, ws=8, dropout=0.0).eval()
    x = torch.randn(1, 32, 64, 64)
    y = x.clone()
    y[..., -2:, -2:] = torch.randn(1, 32, 2, 2) * 3       # (a constant added to every channel would be removed by LayerNorm)
    with torch.no_grad():
        a, b = st(x), st(y)
    assert torch.allclose(a[..., :16, :16], b[..., :16, :16], atol=1e-6)
    assert not torch.allclose(a[..., -8:, -8:], b[..., -8:, -8:], atol=1e-3)
    blk = st.blocks[1]                                           # and without the mask it WOULD leak
    blk.mask = lambda *args: None
    with torch.no_grad():
        assert not torch.allclose(st(x)[..., :4, :4], st(y)[..., :4, :4], atol=1e-6)


def test_padding_is_masked():
    """A map that is not a whole number of windows is padded inside the block; the padding must not act
    as real cells. Far from the edge the result equals that of the same map inside a larger one."""
    torch.manual_seed(0)
    st = nets.SwinStage(32, 2, 4, ws=8, dropout=0.0).eval()
    x = torch.randn(1, 32, 25, 25)
    with torch.no_grad():
        a = st(x)
        b = st(torch.nn.functional.pad(x, (0, 15, 0, 15)))
        assert a.shape == x.shape and torch.allclose(a[..., :4, :4], b[..., :4, :4], atol=1e-6)
        # the padded cells are invisible: changing what the padding WOULD contain cannot matter, so the
        # last real row gives the same answer whether 7 or 15 more rows of padding follow it
        c = st(torch.nn.functional.pad(x, (0, 0, 0, 0)))
        assert torch.allclose(a, c)


def test_trains_and_is_recognised_from_its_weights():
    torch.manual_seed(0)
    m = nets.SwinUNet(12 + 24, 24, 6, width=16)
    loss = m(torch.randn(2, 36, 64, 64)).mean()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    sd = m.state_dict()
    assert any(k.startswith("swin.") for k in sd) and any(k.startswith("temporal.") for k in sd)
    assert not any(k.startswith("bottleneck.") for k in sd) and sd["head.weight"].shape[1] == 16
