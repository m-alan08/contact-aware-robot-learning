"""Contact predictors.

All predictors implement ``predict(X_hist, A_fut) -> (N, H, 4)``.

* PersistencePredictor   -- trivial baseline: future contact = last measured contact.
* RidgePredictor         -- linear action-conditioned model (closed-form ridge).
* MLPPredictor           -- small residual MLP, action-conditioned (main model).
* MLPPredictor(action_conditioned=False) -- same MLP that never sees future actions
  (negative control: its cached and refreshed forecasts are identical by design).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

TAC = 4


def last_tactile(X_hist: np.ndarray, K: int) -> np.ndarray:
    """Last measured tactile frame from flattened history features (layout from
    features.flatten_inputs: tactile (K,4) first)."""
    return X_hist[:, (K - 1) * TAC:K * TAC]


class PersistencePredictor:
    name = "persistence"
    action_conditioned = False

    def __init__(self, K: int, H: int):
        self.K, self.H = K, H

    def fit(self, *a, **k):
        return self

    def predict(self, X_hist, A_fut):
        last = last_tactile(X_hist, self.K)
        return np.repeat(last[:, None, :], self.H, axis=1)


class _Norm:
    def fit(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-6
        return self

    def __call__(self, x):
        return (x - self.mu) / self.sd


class RidgePredictor:
    name = "ridge"
    action_conditioned = True

    def __init__(self, K: int, H: int, alpha: float = 1.0):
        self.K, self.H, self.alpha = K, H, alpha

    def _X(self, X_hist, A_fut):
        return np.concatenate([X_hist, A_fut], 1)

    def fit(self, X_hist, A_fut, Y, **kw):
        X = self._X(X_hist, A_fut)
        self.nx = _Norm().fit(X)
        Xn = np.concatenate([self.nx(X), np.ones((len(X), 1))], 1)
        R = (Y - last_tactile(X_hist, self.K)[:, None, :]).reshape(len(Y), -1)
        A = Xn.T @ Xn + self.alpha * np.eye(Xn.shape[1])
        self.W = np.linalg.solve(A, Xn.T @ R)
        return self

    def predict(self, X_hist, A_fut):
        X = self._X(X_hist, A_fut)
        Xn = np.concatenate([self.nx(X), np.ones((len(X), 1))], 1)
        return last_tactile(X_hist, self.K)[:, None, :] + (Xn @ self.W).reshape(len(X), self.H, TAC)


class _Net(nn.Module):
    def __init__(self, d_in, d_out, hidden, layers):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, d_out))
        self.f = nn.Sequential(*mods)

    def forward(self, x):
        return self.f(x)


class MLPPredictor:
    """Residual MLP: y_hat[t+k] = last measured tactile + f(history, future actions)."""

    def __init__(self, K: int, H: int, action_conditioned: bool = True, hidden: int = 256, layers: int = 3,
                 lr: float = 1e-3, epochs: int = 40, batch: int = 512, weight_decay: float = 1e-5,
                 patience: int = 6, seed: int = 0):
        self.K, self.H = K, H
        self.action_conditioned = action_conditioned
        self.name = "mlp" if action_conditioned else "mlp_history_only"
        self.hp = dict(hidden=hidden, layers=layers, lr=lr, epochs=epochs, batch=batch,
                       weight_decay=weight_decay, patience=patience, seed=seed)
        self.history: list[dict] = []

    def _X(self, X_hist, A_fut):
        return np.concatenate([X_hist, A_fut], 1) if self.action_conditioned else X_hist

    def _resid(self, X_hist, Y):
        return (Y - last_tactile(X_hist, self.K)[:, None, :]).reshape(len(Y), -1)

    def fit(self, X_hist, A_fut, Y, val=None):
        torch.manual_seed(self.hp["seed"])
        X = self._X(X_hist, A_fut)
        R = self._resid(X_hist, Y)
        self.nx = _Norm().fit(X)
        self.ny = _Norm().fit(R)
        self.net = _Net(X.shape[1], R.shape[1], self.hp["hidden"], self.hp["layers"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.hp["epochs"])
        Xt = torch.tensor(self.nx(X), dtype=torch.float32)
        Rt = torch.tensor(self.ny(R), dtype=torch.float32)
        gen = torch.Generator().manual_seed(self.hp["seed"])
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.hp["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            tot = 0.0
            for i in range(0, len(Xt), self.hp["batch"]):
                idx = perm[i:i + self.hp["batch"]]
                loss = ((self.net(Xt[idx]) - Rt[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            rec = {"epoch": ep, "train_mse_norm": tot / len(Xt)}
            if val is not None:
                pv = self.predict(val["X_hist"], val["A_fut"])
                rec["val_rmse_N"] = float(np.sqrt(((pv - val["Y"]) ** 2).mean()))
                if rec["val_rmse_N"] < best - 1e-6:
                    best, bad = rec["val_rmse_N"], 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            self.history.append(rec)
            if val is not None and bad >= self.hp["patience"]:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, X_hist, A_fut):
        self.net.eval()
        X = torch.tensor(self.nx(self._X(X_hist, A_fut)), dtype=torch.float32)
        r = self.net(X).numpy() * self.ny.sd + self.ny.mu
        return last_tactile(X_hist, self.K)[:, None, :] + r.reshape(len(X), self.H, TAC)

    def state(self):
        return {"hp": self.hp, "K": self.K, "H": self.H, "action_conditioned": self.action_conditioned,
                "nx": (self.nx.mu, self.nx.sd), "ny": (self.ny.mu, self.ny.sd), "net": self.net.state_dict()}

    @classmethod
    def from_state(cls, s):
        m = cls(s["K"], s["H"], s["action_conditioned"], **{k: v for k, v in s["hp"].items()})
        m.nx, m.ny = _Norm(), _Norm()
        m.nx.mu, m.nx.sd = s["nx"]
        m.ny.mu, m.ny.sd = s["ny"]
        d_in = len(m.nx.mu)
        m.net = _Net(d_in, len(m.ny.mu), s["hp"]["hidden"], s["hp"]["layers"])
        m.net.load_state_dict(s["net"])
        return m


def timed_predict(model, X_hist, A_fut, repeats: int = 1):
    """Returns (prediction, seconds per call) for single-sample inference latency."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        y = model.predict(X_hist, A_fut)
    return y, (time.perf_counter() - t0) / repeats
