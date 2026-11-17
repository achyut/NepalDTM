"""Phase 4 (G): conditional diffusion model, 8 m M -> 1 m terrain, with the 8 m consistency projection.

Data: clean 1 m lidar patches (g/make_patches.py); the 8 m condition is made on the fly by g/error_model.degrade,
so G learns from inputs that carry the measured errors of the Nepal product (the EXP026 lesson).
Target: the 1 m residual from the bicubic upsampling of the 8 m input, in units of RSCALE metres.
Model: UNet with time embedding, v-prediction, cosine schedule (Ho et al. 2020; Nichol & Dhariwal 2021;
Salimans & Ho 2022); DDIM sampling (Song et al. 2021).
Projection (plan): G - blockmean8(G) + M, so G averaged to 8 m equals M exactly.
Holdout: every 20th patch (5%), never trained on.

Run:
  ../.venv/bin/python -I -u g/g_model.py prep              # patches -> one int16 memmap
  ../.venv/bin/python -I -u g/g_model.py train [--steps N]  # resumable, checkpoints every 2k steps
  ../.venv/bin/python -I -u g/g_model.py eval  [--n 256]    # held-out: bicubic vs G (1 sample, ensemble median)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "g"))
from error_model import OUT as EM_PATH, degrade  # noqa: E402

GEN = ROOT / "data" / "gen"
MM = GEN / "patches_all.int16"
META = GEN / "patches_meta.npz"
CK = ROOT / "data" / "checkpoints" / "g_model"
RSCALE = 2.0  # metres per unit of the residual target
T_STEPS = 1000
PROJECT_ITERS = 8  # smooth consistency projection: residual block-mean error shrinks ~geometrically per pass
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------------- data
def prep():
    shards = sorted((GEN / "patches").glob("*.npz"))
    n = sum(len(np.load(p)["zmin"]) for p in shards)
    mm = np.memmap(MM, "int16", "w+", shape=(n, 256, 256))
    meta = {k: [] for k in ("zmin", "scale", "slope_mean", "relief", "src")}
    i = 0
    for p in shards:
        d = np.load(p)
        k = len(d["zmin"])
        mm[i:i + k] = d["z"]
        for key in meta:
            meta[key].append(d[key])
        i += k
        print(f"  {p.name}: {i:,}/{n:,}", flush=True)
    mm.flush()
    np.savez(META, n=n, **{k: np.concatenate(v) for k, v in meta.items()})
    print(f"{n:,} patches -> {MM}")


class Patches(torch.utils.data.Dataset):
    def __init__(self, split: str):
        m = np.load(META)
        self.n = int(m["n"])
        self.zmin, self.scale = m["zmin"], m["scale"]
        idx = np.arange(self.n)
        self.idx = idx[idx % 20 == 0] if split == "val" else idx[idx % 20 != 0]
        self.model = json.loads(EM_PATH.read_text())
        self.mm = None
        self.split = split

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        if self.mm is None:
            self.mm = np.memmap(MM, "int16", "r", shape=(self.n, 256, 256))
        i = int(self.idx[j])
        z = self.mm[i].astype("float32") * self.scale[i] / 100
        rng = np.random.default_rng(i if self.split == "val" else None)
        if self.split == "train":  # rotations and flips
            z = np.rot90(z, rng.integers(4)).copy()
            if rng.random() < 0.5:
                z = z[:, ::-1].copy()
        m8 = degrade(z, self.model, rng=rng)
        return torch.from_numpy(z)[None], torch.from_numpy(m8)[None]


def condition(m8: torch.Tensor):
    """8 m input -> (bicubic 1 m upsampling, normalized condition channels, offset)."""
    off = m8.mean((2, 3), keepdim=True)
    up = F.interpolate(m8 - off, scale_factor=8, mode="bicubic", align_corners=False)
    gy, gx = torch.gradient(up, dim=(2, 3))
    cond = torch.cat([up / 50.0, gx / 0.5, gy / 0.5], 1)  # relief in 50 m units, gradients per 1 m px
    return up, cond, off


# ----------------------------------------------------------------------------- model
def temb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Res(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(te)[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class UNet(nn.Module):
    def __init__(self, cin=4, ch=(64, 128, 192, 256), tdim=256):
        super().__init__()
        self.tdim = tdim
        self.tm = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(cin, ch[0], 3, padding=1)
        self.down = nn.ModuleList()
        c = ch[0]
        skips = []
        for i, co in enumerate(ch):
            self.down.append(nn.ModuleList([Res(c, co, tdim), Res(co, co, tdim),
                                            nn.Conv2d(co, co, 3, stride=2, padding=1) if i < len(ch) - 1 else nn.Identity()]))
            skips.append(co)
            c = co
        self.mid = nn.ModuleList([Res(c, c, tdim), Res(c, c, tdim)])
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            self.up.append(nn.ModuleList([Res(c + skips[i], co, tdim), Res(co, co, tdim),
                                          nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity()]))
            c = co
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, cond):
        te = self.tm(temb(t, self.tdim))
        h = self.inp(torch.cat([x, cond], 1))
        hs = []
        for r1, r2, dn in self.down:
            h = r2(r1(h, te), te)
            hs.append(h)
            h = dn(h)
        for r in self.mid:
            h = r(h, te)
        for (r1, r2, upm), s in zip(self.up, reversed(hs)):
            h = r2(r1(torch.cat([h, s], 1), te), te)
            h = upm(h)
        return self.out(h)


def alpha_bar(t):
    return torch.cos((t / T_STEPS + 0.008) / 1.008 * math.pi / 2) ** 2


# ----------------------------------------------------------------------------- train / sample
def train(steps: int, bs: int):
    CK.mkdir(parents=True, exist_ok=True)
    net = UNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    ema = UNet().to(DEV)
    ema.load_state_dict(net.state_dict())
    step = 0
    last = CK / "last.pt"
    if last.exists():
        s = torch.load(last, map_location=DEV)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step}", flush=True)
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M, device {DEV}", flush=True)
    dl = torch.utils.data.DataLoader(Patches("train"), batch_size=bs, shuffle=True, num_workers=6,
                                     persistent_workers=True, drop_last=True)
    t0, run = time.time(), 0.0
    while step < steps:
        for z, m8 in dl:
            z, m8 = z.to(DEV), m8.to(DEV)
            up, cond, off = condition(m8)
            x0 = (z - off - up) / RSCALE
            t = torch.randint(1, T_STEPS + 1, (z.shape[0],), device=DEV)
            ab = alpha_bar(t.float())[:, None, None, None]
            eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            v = ab.sqrt() * eps - (1 - ab).sqrt() * x0
            loss = F.mse_loss(net(xt, t, cond), v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = 2e-4 * min(1.0, (step + 1) / 2000)
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(0.999).add_(pn, alpha=0.001)
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 100 == 0:
                print(f"step {step}  loss {run:.4f}  {(time.time() - t0) / 100:.2f} s/step", flush=True)
                t0 = time.time()
            if step % 2000 == 0 or step == steps:
                torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}, last)
                torch.save({"ema": ema.state_dict(), "step": step}, CK / f"ema_{step:06d}.pt")
            if step >= steps:
                break


@torch.no_grad()
def sample(net, m8: torch.Tensor, n_steps=50, eta=0.0, project=True):
    """DDIM sampling; returns 1 m heights (B,1,256,256). project: G averages exactly to M at 8 m."""
    up, cond, off = condition(m8)
    x = torch.randn_like(up)
    ts = torch.linspace(T_STEPS, 1, n_steps + 1, device=m8.device)
    for a, b in zip(ts[:-1], ts[1:]):
        aa, ab_ = alpha_bar(a), alpha_bar(b)
        v = net(x, a.repeat(x.shape[0]).long(), cond)
        x0 = aa.sqrt() * x - (1 - aa).sqrt() * v
        e = (1 - aa).sqrt() * x + aa.sqrt() * v
        x = ab_.sqrt() * x0 + (1 - ab_).sqrt() * e
    z = up + off + RSCALE * x0
    if project:
        # Make every 8 x 8 block average to M without leaving steps at the block edges: the correction is applied as
        # a smooth (bicubic) surface and repeated, so block means converge to M. A per-block constant shift would
        # match the means exactly but print an 8 m grid into the 1 m surface.
        for _ in range(PROJECT_ITERS):
            z = z + F.interpolate(m8 - F.avg_pool2d(z, 8), scale_factor=8, mode="bicubic", align_corners=False)
    return z


def evaluate(n: int, ens: int, ckpt: str | None):
    p = Path(ckpt) if ckpt else sorted(CK.glob("ema_*.pt"))[-1]
    net = UNet().to(DEV)
    s = torch.load(p, map_location=DEV)
    net.load_state_dict(s["ema"]); net.eval()
    ds = Patches("val")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    E = {k: [] for k in ("input_8m_nearest", "bicubic", "G_1_sample", f"G_median_{ens}")}
    G = {k: [] for k in E}
    for j in idx:
        z, m8 = ds[j]
        z, m8 = z[None].to(DEV), m8[None].to(DEV)
        up, _, off = condition(m8)
        cands = {"input_8m_nearest": F.interpolate(m8, scale_factor=8, mode="nearest"), "bicubic": up + off}
        S = sample(net, m8.repeat(ens, 1, 1, 1))
        cands["G_1_sample"], cands[f"G_median_{ens}"] = S[:1], S.median(0, keepdim=True).values
        for k, c in cands.items():
            e = (c - z)[0, 0].cpu().numpy()
            E[k].append(e.ravel())
            gz = np.hypot(*np.gradient(z[0, 0].cpu().numpy())); gc = np.hypot(*np.gradient(c[0, 0].cpu().numpy()))
            G[k].append(float(np.corrcoef(gz.ravel(), gc.ravel())[0, 1]))
    res = {}
    for k in E:
        e = np.concatenate(E[k])
        res[k] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "nmad": float(1.4826 * np.median(np.abs(e - np.median(e)))),
                  "slope_corr": float(np.nanmean(G[k]))}
        print(f"  {k:18s} RMSE {res[k]['rmse']:.3f}  NMAD {res[k]['nmad']:.3f}  slope-map corr {res[k]['slope_corr']:.3f}")
    out = CK / f"eval_{s['step']:06d}.json"
    out.write_text(json.dumps({"ckpt": str(p), "n": len(idx), "results": res}, indent=2))
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "train", "eval"))
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ens", type=int, default=8)
    ap.add_argument("--ckpt")
    a = ap.parse_args()
    {"prep": prep, "train": lambda: train(a.steps, a.bs), "eval": lambda: evaluate(a.n, a.ens, a.ckpt)}[a.cmd]()
