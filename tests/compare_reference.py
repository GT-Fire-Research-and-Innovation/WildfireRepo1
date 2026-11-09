"""Compare physics.rothermel against pyretechnics (run with REF=path/to/pyretechnics/src)."""
import os, sys, random
sys.path.insert(0, os.environ["REF"]); sys.path.insert(0, ".")
import torch
import pyretechnics.fuel_models as fm, pyretechnics.surface_fire as sf
from physics import rothermel as R
random.seed(0)
rows = []
codes = [c for c in R.CODES if c > 100]
for _ in range(400):
    c = random.choice(codes)
    m = [random.uniform(.03, .14), random.uniform(.04, .16), random.uniform(.05, .2), 0, random.uniform(.3, 1.5), random.uniform(.6, 1.6)]
    m[3] = m[0]
    ws = random.uniform(0, 12); wd = random.uniform(0, 360); sl = random.uniform(0, 0.8); asp = random.uniform(0, 360)
    cc, ch = random.choice([(0, 0), (random.uniform(.1, .8), random.uniform(3, 30))])
    rows.append((c, m, ws, wd, sl, asp, cc, ch))
# reference
ref = []
for c, m, ws, wd, sl, asp, cc, ch in rows:
    fmod = fm.moisturize(fm.fuel_model_table[c], tuple(m))
    mn = sf.calc_surface_fire_behavior_no_wind_no_slope(fmod)
    u20 = ws / 1.15 * 60
    mid = sf.calc_midflame_wind_speed(u20, fmod.delta, ch / 0.3048, cc)
    mx = sf.calc_surface_fire_behavior_max(mn, mid, wd, sl, asp, True, "behave")
    d = mx.max_spread_direction
    ref.append((mn.base_spread_rate, mx.max_spread_rate, mx.length_to_width_ratio, d[0], d[1]))
t = lambda i: torch.tensor([r[i] for r in rows], dtype=torch.float64)
idx = R.fuel_index(torch.tensor([r[0] for r in rows]))
M = torch.tensor([r[1] for r in rows], dtype=torch.float64)
out = R.spread(idx, M[:, 0], M[:, 1], M[:, 2], M[:, 4], M[:, 5], t(2), t(3), t(4), t(5), t(6), t(7))
ref = torch.tensor(ref, dtype=torch.float64)
def rel(a, b): return ((a - b).abs() / (b.abs() + 1e-6)).max().item()
print("r0 max rel err     ", rel(out["r0"], ref[:, 0]))
print("r_head max rel err ", rel(out["r_head"], ref[:, 1]))
print("L/B max abs err    ", (out["lb"] - ref[:, 2]).abs().max().item())
hd = torch.stack([ref[:, 3], ref[:, 4]], 1); hd = hd / hd.norm(dim=1, keepdim=True)
cosang = out["head_x"] * hd[:, 0] + out["head_y"] * hd[:, 1]
mask = ref[:, 1] > 0
print("heading min cos    ", cosang[mask].min().item())
print("r_head range m/min ", ref[:, 1].min().item(), ref[:, 1].max().item())
bad = ((out["r_head"] - ref[:, 1]).abs() / (ref[:, 1] + 1e-6)) > 1e-3
print("cases >0.1% off:", int(bad.sum()), [rows[i][0] for i in torch.where(bad)[0][:10].tolist()])
