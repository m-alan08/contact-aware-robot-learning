"""Small MLPs: trained with PyTorch, executed with NumPy for fast single-sample
closed-loop inference (no autograd / dispatcher overhead per control step)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.special import erf as _erf
from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, layers: int = 3, out_act: str = "none"):
        super().__init__()
        mods: List[nn.Module] = []
        d = d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        mods.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*mods)
        self.out_act = out_act

    def forward(self, x):
        y = self.net(x)
        return torch.sigmoid(y) if self.out_act == "sigmoid" else y


def train_mlp(X: np.ndarray, Y: np.ndarray, Xv: np.ndarray, Yv: np.ndarray, *, seed: int, hidden=256, layers=3,
              out_act="none", epochs=30, batch=512, lr=1e-3, wd=1e-5, log=None) -> Dict:
    """Fit an MLP with input/output standardisation. The final epoch's weights are
    kept (no checkpoint selection on any outcome); validation loss is logged only."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    if out_act == "sigmoid":
        ymu, ysd = np.zeros(Y.shape[1]), np.ones(Y.shape[1])
    else:
        ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    net = MLP(X.shape[1], Y.shape[1], hidden, layers, out_act)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Y - ymu) / ysd, dtype=torch.float32)
    Xvt = torch.tensor((Xv - mu) / sd, dtype=torch.float32)
    Yvt = torch.tensor((Yv - ymu) / ysd, dtype=torch.float32)
    hist = []
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(Xvt) - Yvt) ** 2).mean().item()
        hist.append((ep, tot / len(Xt), vl))
        if log:
            log(f"  epoch {ep:2d} train {tot / len(Xt):.4f} val {vl:.4f}")
    W = [(m.weight.detach().numpy().astype(np.float64).copy(), m.bias.detach().numpy().astype(np.float64).copy())
         for m in net.net if isinstance(m, nn.Linear)]
    return {"W": W, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd, "out_act": out_act, "history": hist}


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))


class NumpyMLP:
    """Inference copy of a trained MLP (exact GELU; matches PyTorch to float precision)."""

    def __init__(self, state: Dict):
        self.W = state["W"]
        self.mu, self.sd, self.ymu, self.ysd = state["mu"], state["sd"], state["ymu"], state["ysd"]
        self.out_act = state["out_act"]

    def forward(self, x: np.ndarray, return_hidden: bool = False):
        h = (x - self.mu) / self.sd
        for W, b in self.W[:-1]:
            h = _gelu(h @ W.T + b)
        y = h @ self.W[-1][0].T + self.W[-1][1]
        if self.out_act == "sigmoid":
            y = 1.0 / (1.0 + np.exp(-y))
        y = y * self.ysd + self.ymu
        return (y, h) if return_hidden else y

    __call__ = forward
