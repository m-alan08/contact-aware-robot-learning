"""TacPAC-structured tactile forecast + corrector heads for H2-001.

This is NOT TacPAC. It reproduces TacPAC's *interface* (a frozen base policy
plans an action chunk; a cached prediction of the contact that chunk will
produce is combined with each newly measured tactile frame to correct the
unexecuted suffix) at a size that trains on one GPU, on top of a ManiFeel
diffusion policy. See docs/experiments/H2-001_BASELINE_SELECTION.md §6.

Design choices that make the H2 interventions well defined:

* The forecast is EXPLICIT: future tactile force fields (C, H, W) for each
  step of the chunk, supervised by the demonstrations' measured future.
* The corrector receives three pathways as SEPARATE token groups, each with its
  own input projection:
      context  -- frozen base-policy observation features (visual + state),
      live     -- the tactile frame measured now,
      forecast -- the cached (or intervened) predicted tactile frames,
  plus action tokens (planned chunk, with an executed/unexecuted flag) and an
  execution-offset token. An intervention on one pathway therefore leaves the
  other pathways' inputs bit-identical.
* Forecast dropout: during training the forecast tokens are replaced by a
  learned null token with probability ``p_null``. "Disable the forecast"
  (condition C) is then an input the corrector was trained on, not an
  out-of-distribution zero tensor.
* A ``use_forecast=False`` corrector (condition D: live touch, no prediction)
  has the same architecture minus the forecast tokens.

Compatible with Python 3.8 (ManiFeel's environment).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

TAC_SHAPE_DEFAULT = (3, 10, 14)  # ManiFeel TacSL force field: normal + 2 shear, 10x14 taxels


def _mlp(i: int, h: int, o: int, depth: int = 2) -> nn.Sequential:
    layers = []
    d = i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def _sinusoid(x: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = x.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TactileForecaster(nn.Module):
    """Predicts the tactile frames an action chunk will produce.

    Inputs: base-policy context features (B, Dc), recent measured tactile
    frames (B, To, C, H, W), planned action chunk (B, T, Da).
    Output: predicted tactile frames for each chunk step (B, T, C, H, W),
    parameterised as a residual on the last measured frame.
    """

    def __init__(self, context_dim: int, action_dim: int, horizon: int,
                 tac_shape: Sequence[int] = TAC_SHAPE_DEFAULT, n_obs_steps: int = 2,
                 hidden: int = 512):
        super().__init__()
        self.tac_shape = tuple(tac_shape)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        tdim = int(torch.tensor(self.tac_shape).prod())
        self.tdim = tdim
        self.net = _mlp(context_dim + n_obs_steps * tdim + horizon * action_dim, hidden,
                        horizon * tdim, depth=3)

    def forward(self, context: torch.Tensor, tac_hist: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        x = torch.cat([context, tac_hist.reshape(B, -1), actions.reshape(B, -1)], dim=-1)
        delta = self.net(x).reshape(B, self.horizon, *self.tac_shape)
        return tac_hist[:, -1:].expand(-1, self.horizon, *self.tac_shape) + delta


class TactileCorrector(nn.Module):
    """Single-pass corrector: delta actions for the unexecuted suffix of a chunk."""

    def __init__(self, context_dim: int, action_dim: int, horizon: int,
                 tac_shape: Sequence[int] = TAC_SHAPE_DEFAULT, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, use_forecast: bool = True,
                 p_null: float = 0.2):
        super().__init__()
        self.tac_shape = tuple(tac_shape)
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_model
        self.use_forecast = use_forecast
        self.p_null = p_null
        tdim = int(torch.tensor(self.tac_shape).prod())
        self.context_proj = _mlp(context_dim, d_model, d_model)
        self.live_proj = _mlp(tdim, d_model, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.executed_emb = nn.Embedding(2, d_model)
        self.step_emb = nn.Embedding(horizon, d_model)
        self.offset_proj = _mlp(d_model, d_model, d_model)
        # token-type embeddings: 0 context, 1 live, 2 forecast, 3 action, 4 offset
        self.type_emb = nn.Embedding(5, d_model)
        if use_forecast:
            self.forecast_proj = _mlp(tdim, d_model, d_model)
            self.null_forecast = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.head.weight)  # start as identity correction (delta = 0), as TacPAC
        nn.init.zeros_(self.head.bias)
        self._last_hidden: Optional[torch.Tensor] = None

    def forecast_tokens(self, forecast: Optional[torch.Tensor], B: int,
                        null_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Forecast pathway tokens (B, T, d). ``forecast=None`` -> all null tokens.
        ``null_mask`` (B,) bool replaces the whole forecast of those rows by null."""
        null = self.null_forecast.expand(B, self.horizon, -1)
        if forecast is None:
            return null
        tok = self.forecast_proj(forecast.reshape(B, self.horizon, -1))
        if null_mask is not None:
            tok = torch.where(null_mask[:, None, None], null, tok)
        return tok

    def forward(self, context: torch.Tensor, live_tac: torch.Tensor, actions: torch.Tensor,
                offset: torch.Tensor, forecast: Optional[torch.Tensor] = None,
                forecast_scale: float = 1.0, null_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        context:  (B, Dc) base-policy features at plan time (or refreshed, for the matched control)
        live_tac: (B, C, H, W) tactile frame measured now
        actions:  (B, T, Da) executed prefix + planned suffix (normalised)
        offset:   (B,) long, number of executed steps k (0 <= k < T)
        forecast: (B, T, C, H, W) predicted tactile frames, or None (= null forecast)
        forecast_scale: multiplies forecast tokens (graded influence; 1.0 = trained setting)
        Returns delta actions (B, T, Da); rows < offset are zeroed (already executed).
        """
        B, T, _ = actions.shape
        assert T == self.horizon
        dev = actions.device
        steps = torch.arange(T, device=dev)
        executed = steps[None, :] < offset[:, None]
        tok_ctx = self.context_proj(context)[:, None] + self.type_emb.weight[0]
        tok_live = self.live_proj(live_tac.reshape(B, -1))[:, None] + self.type_emb.weight[1]
        tok_act = (self.action_proj(actions) + self.executed_emb(executed.long())
                   + self.step_emb(steps)[None] + self.type_emb.weight[3])
        tok_off = self.offset_proj(_sinusoid(offset, self.d_model))[:, None] + self.type_emb.weight[4]
        groups = [tok_ctx, tok_live]
        if self.use_forecast:
            tok_fc = self.forecast_tokens(forecast, B, null_mask) * forecast_scale
            groups.append(tok_fc + self.step_emb(steps)[None] + self.type_emb.weight[2])
        groups += [tok_off, tok_act]
        x = self.encoder(torch.cat(groups, dim=1))
        h_act = x[:, -T:]
        self._last_hidden = h_act.detach()
        delta = self.head(h_act)
        return delta * (~executed)[..., None].to(delta.dtype)

    def training_null_mask(self, B: int, device) -> Optional[torch.Tensor]:
        if not (self.training and self.use_forecast and self.p_null > 0):
            return None
        return torch.rand(B, device=device) < self.p_null


# --------------------------------------------------------------------------
# Interventions on the forecast pathway (H2 conditions and matched controls)
# --------------------------------------------------------------------------

INTERVENTIONS = (
    "cached",          # A: forecast made at plan time, kept for the whole chunk
    "refresh",         # B: forecast recomputed at the current step from current obs
    "null",            # C: forecast pathway replaced by the trained null token
    "scale",           # C': graded reduction, forecast tokens * alpha
    "shuffle",         # E: a cached forecast from ANOTHER trial (in-distribution, wrong content)
    "norm_matched",    # E: cached forecast + random perturbation with the same norm as (refresh - cached)
    "context_refresh", # E: refresh the visual/state context only, keep the cached forecast
    "oracle",          # O: measured future tactile frames (NON-DEPLOYABLE diagnostic)
)


def apply_intervention(mode: str, cached: torch.Tensor, *, refreshed: Optional[torch.Tensor] = None,
                       other_trial: Optional[torch.Tensor] = None, oracle: Optional[torch.Tensor] = None,
                       generator: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
    """Return the forecast tensor the corrector should see under ``mode``.

    ``None`` means the null forecast. ``scale`` and ``context_refresh`` return the
    cached forecast unchanged; the caller applies ``forecast_scale`` or swaps the
    context respectively.
    """
    if mode in ("cached", "scale", "context_refresh"):
        return cached
    if mode == "refresh":
        assert refreshed is not None
        return refreshed
    if mode == "null":
        return None
    if mode == "shuffle":
        assert other_trial is not None and other_trial.shape == cached.shape
        return other_trial
    if mode == "norm_matched":
        assert refreshed is not None
        diff = refreshed - cached
        noise = torch.randn(cached.shape, generator=generator, device=cached.device, dtype=cached.dtype)
        dn = diff.flatten(1).norm(dim=1)
        nn_ = noise.flatten(1).norm(dim=1).clamp_min(1e-12)
        noise = noise * (dn / nn_).view(-1, *([1] * (cached.dim() - 1)))
        return cached + noise
    if mode == "oracle":
        assert oracle is not None
        return oracle
    raise ValueError("unknown intervention: %s" % mode)


# --------------------------------------------------------------------------
# Training targets (TacPAC stage-2 analogue)
# --------------------------------------------------------------------------

def corrector_target(plan: torch.Tensor, demo: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Delta-action target and loss mask for the unexecuted suffix.

    plan, demo: (B, T, Da) frozen-base plan and demonstrated actions for the same
    chunk; offset: (B,) executed steps. Target = demo - plan on steps >= offset.
    """
    T = plan.shape[1]
    mask = torch.arange(T, device=plan.device)[None, :] >= offset[:, None]
    return (demo - plan) * mask[..., None], mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[..., None].to(pred.dtype)
    return ((pred - target) ** 2 * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)


# --------------------------------------------------------------------------
# OOD check for interventions (plan §4.2: swaps can create unnatural states)
# --------------------------------------------------------------------------

class ActivationStats:
    """Mean/covariance of corrector action-token hidden states under nominal
    inputs; Mahalanobis distance flags out-of-distribution intervention states."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.prec: Optional[torch.Tensor] = None

    def fit(self, hidden: torch.Tensor) -> "ActivationStats":
        h = hidden.reshape(-1, hidden.shape[-1]).double()
        self.mean = h.mean(0)
        cov = torch.cov(h.T) + self.eps * torch.eye(h.shape[1], dtype=h.dtype)
        self.prec = torch.linalg.inv(cov)
        return self

    def distance(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None
        h = hidden.reshape(-1, hidden.shape[-1]).double() - self.mean
        return torch.sqrt(torch.einsum("nd,de,ne->n", h, self.prec, h).clamp_min(0))


def action_change(delta_a: torch.Tensor, delta_b: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Per-trial L2 norm of the difference between two corrections on the unexecuted suffix
    (the 'does the outdated forecast change the selected action?' metric)."""
    T = delta_a.shape[1]
    mask = (torch.arange(T, device=delta_a.device)[None, :] >= offset[:, None])[..., None]
    return ((delta_a - delta_b) * mask).flatten(1).norm(dim=1)


def cost_record(n_corrector_calls: int, n_forecaster_calls: int, n_base_calls: int) -> Dict[str, int]:
    """Inference-budget bookkeeping so conditions can be matched on compute."""
    return {"corrector": n_corrector_calls, "forecaster": n_forecaster_calls, "base": n_base_calls}
