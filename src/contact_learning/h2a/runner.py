"""Closed-loop episodes for H2-001A: features, forecast caching/refresh rules,
controllers, branching at disturbance onset, and outcome metrics.

Timing convention (sensing latency L control steps):
  obs[u] is the observation after u control steps (obs[0] = after reset).
  At decision time t the controller may read tactile obs[tau], tau = max(0, t - L),
  and proprioception obs[t] (joint encoders are not delayed).
  A forecast made at decision time c predicts tactile for times tau_c+1 ... tau_c+HF,
  assuming the grip commands already sent in [tau_c, c) and then a held grip,
  with the plan's lift commands.
  The forecast slice given to the policy at time t covers tau+1 ... tau+HP.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from contact_learning.h2a.env import DisturbedLiftEnv, Disturbance, grip_for_normal
from contact_learning.h2a.expert import CRUSH_SAFETY, disturbance_for_severity, expert_grip
from contact_learning.h2a.nets import NumpyMLP

K = 4        # tactile / proprio history length
HF = 20      # forecaster horizon
HP = 9       # forecast slice length seen by the policy
CHUNK = 10   # deployed cache refresh period (steps)
TAC = 4

REFRESH_MODES = ("chunk", "every", "null", "fixed3", "gate", "gate_null", "random", "oracle")


# ---------------------------------------------------------------- episode sampling
@dataclass
class EpisodeSpec:
    seed: int
    kind: str = "none"
    severity: float = float("nan")
    onset_frac: float = 0.5

    def disturbance(self, env: DisturbedLiftEnv) -> Disturbance:
        if self.kind == "none":
            return Disturbance()
        lift_end = int(np.nonzero(env.lift_cmd)[0].max())
        lo, hi = lift_end + 3, env.h2.T - 25
        onset = int(lo + round((hi - lo) * self.onset_frac))
        return disturbance_for_severity(env, self.kind, self.severity, onset)

    def nominal_branch_step(self, env: DisturbedLiftEnv) -> int:
        lift_end = int(np.nonzero(env.lift_cmd)[0].max())
        lo, hi = lift_end + 3, env.h2.T - 25
        return int(lo + round((hi - lo) * self.onset_frac))


def sample_spec(seed: int, kinds, severity_range, p_nominal: float) -> EpisodeSpec:
    rng = np.random.default_rng([seed, 4242])
    of = float(rng.uniform())
    if rng.uniform() < p_nominal:
        return EpisodeSpec(seed, "none", float("nan"), of)
    return EpisodeSpec(seed, str(rng.choice(list(kinds))), float(rng.uniform(*severity_range)), of)


def start_episode(env: DisturbedLiftEnv, spec: EpisodeSpec) -> dict:
    env.reset_episode(spec.seed)
    d = spec.disturbance(env)
    return env.reset_episode(spec.seed, d)


# ---------------------------------------------------------------- features
def tac_hist(tac: List[np.ndarray], tau: int) -> np.ndarray:
    idx = np.clip(np.arange(tau - K + 1, tau + 1), 0, None)
    return np.concatenate([tac[i] for i in idx])


def forecaster_input(tac, prop, grip_cmds, lift_cmd, tau: int, t: int, g_hold: float) -> np.ndarray:
    """History at tau, executed grip commands for [tau, t), then a held grip; lift plan."""
    idx = np.clip(np.arange(tau - K + 1, tau + 1), 0, None)
    ph = np.concatenate([prop[i] for i in idx])
    gseq = np.full(HF, g_hold)
    ex = grip_cmds[tau:t]
    gseq[: len(ex)] = ex[:HF]
    lseq = np.array([lift_cmd[u] if u < len(lift_cmd) else 0.0 for u in range(tau, tau + HF)])
    return np.concatenate([tac_hist(tac, tau), ph, gseq, lseq])


def context_feat(env: DisturbedLiftEnv, t: int) -> np.ndarray:
    lc = env.lift_cmd
    T = env.h2.T
    return np.array([env.crush_limit, env.n_static_est, float(t >= env.plan.close_start), lc[t] if t < T else 0.0,
                     lc[t - 1] if 0 < t <= T else 0.0, t / T])


def policy_input(env, tac, prop, t, tau, forecast_slice: Optional[np.ndarray], age: int, use_forecast: bool):
    base = np.concatenate([tac_hist(tac, tau), prop[t], context_feat(env, t)])
    if not use_forecast:
        return base
    if forecast_slice is None:
        fpart = np.zeros(HP * TAC + 2)
        fpart[-1] = 1.0  # null flag
    else:
        fpart = np.concatenate([forecast_slice.ravel(), [age / CHUNK, 0.0]])
    return np.concatenate([base, fpart])


# ---------------------------------------------------------------- controllers
@dataclass
class Models:
    forecaster: Optional[NumpyMLP] = None
    policy_p: Optional[NumpyMLP] = None   # forecast-conditioned student
    policy_d: Optional[NumpyMLP] = None   # live-feedback-only student


@dataclass
class ReactiveParams:
    ratio: float = 0.4    # target tangential/normal ratio (slip margin); tuned on dev seeds
    prior_gain: float = 1.4  # floor = prior_gain x weight-based estimate of the static requirement


@dataclass
class RunConfig:
    agent: str = "P"              # P | D | R | X
    refresh: str = "chunk"        # see REFRESH_MODES (P only)
    theta: float = np.inf         # discrepancy threshold for gate modes
    random_steps: frozenset = frozenset()
    exec_noise: float = 0.0       # training-data exploration noise on executed grip
    beta: float = 0.0             # probability of executing the expert action (DAgger)
    record_features: bool = False
    p_null_record: float = 0.0    # forecast dropout applied to recorded P features
    action_probe: bool = False    # paired action-level interventions at each step
    probe_window: tuple = (0, 0)
    record_hidden: bool = False


class Episode:
    """One episode's controller state. ``snapshot``/``restore`` allow branching."""

    def __init__(self, env: DisturbedLiftEnv, models: Models, obs0: dict, latency: int, rng_seed: int):
        self.env, self.m, self.L = env, models, latency
        self.tac = [obs0["tactile"].copy()]
        self.tac_clean = [obs0["tactile_clean"].copy()]
        self.prop = [obs0["proprio"].copy()]
        self.grip_cmds: List[float] = []
        self.rel = []
        self.cache = None          # (HF, 4)
        self.cache_tau = -1
        self.cache_c = -1
        self.null_until = -1       # gate_null: suppress until this step
        self.t = 0
        self.rng = np.random.default_rng([rng_seed, 99])
        self.log: Dict[str, list] = {k: [] for k in (
            "g", "g_star", "refresh", "disc", "pred_now", "fc_err", "n_min", "forecaster_calls", "x", "y",
            "probe", "hidden")}
        self.forecaster_calls = 0
        self.crush_run = 0
        self.crushed = False

    # -- state
    def snapshot(self):
        st = {k: copy.deepcopy(getattr(self, k)) for k in (
            "tac", "tac_clean", "prop", "grip_cmds", "rel", "cache", "cache_tau", "cache_c", "null_until", "t",
            "log", "forecaster_calls", "crush_run", "crushed")}
        st["rng"] = copy.deepcopy(self.rng.bit_generator.state)
        st["env"] = self.env.get_state()
        return st

    def restore(self, st):
        for k, v in st.items():
            if k in ("rng", "env"):
                continue
            setattr(self, k, copy.deepcopy(v))
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(st["rng"])
        self.env.set_state(st["env"])

    # -- forecasting
    def tau(self, t=None):
        t = self.t if t is None else t
        return max(0, t - self.L)

    def make_forecast(self) -> np.ndarray:
        t, tau = self.t, self.tau()
        x = forecaster_input(self.tac, self.prop, np.asarray(self.grip_cmds), self.env.lift_cmd, tau, t,
                             self.env.grip_applied)
        self.forecaster_calls += 1
        return self.tac[tau] + self.m.forecaster(x).reshape(HF, TAC)

    def oracle_forecast(self) -> np.ndarray:
        """True future pad forces under the held-grip assumption, from the current
        simulator state (NON-DEPLOYABLE diagnostic). Index 0 predicts tau+1."""
        st = self.env.get_state()
        g = self.env.grip_applied
        seq = [self.tac_clean[u] for u in range(self.tau() + 1, self.t + 1)]
        while len(seq) < HF:
            o = self.env.step_grip(g)
            seq.append(o["tactile_clean"].copy())
        self.env.set_state(st)
        self.forecaster_calls += 1
        return np.array(seq[:HF])

    def set_cache(self, f):
        self.cache, self.cache_tau, self.cache_c = f, self.tau(), self.t

    def slice(self):
        if self.cache is None:
            return None, 0
        o = self.tau() - self.cache_tau
        s = self.cache[o:o + HP]
        if len(s) < HP:
            s = np.vstack([s, np.repeat(s[-1:], HP - len(s), 0)])
        return s, self.t - self.cache_c

    def discrepancy(self) -> float:
        if self.cache is None:
            return 0.0
        i = self.tau() - self.cache_tau - 1
        if i < 0 or i >= len(self.cache):
            return 0.0
        return float(np.abs(self.tac[self.tau()] - self.cache[i]).mean())

    def update_forecast(self, rc: RunConfig) -> tuple:
        """Apply the refresh rule for this step; returns (slice, age, refreshed, disc)."""
        t = self.t
        mode = rc.refresh
        refreshed = False
        if mode == "null":
            return None, 0, False, 0.0
        disc = self.discrepancy()
        if mode == "oracle":
            self.set_cache(self.oracle_forecast())
            refreshed = True
        else:
            due = (t % CHUNK == 0) or self.cache is None
            if mode == "every":
                due = True
            elif mode == "fixed3":
                due = (t % 3 == 0) or self.cache is None
            elif mode in ("gate", "gate_null"):
                if disc > rc.theta and not due:
                    if mode == "gate":
                        due = True
                    else:
                        self.null_until = (t // CHUNK + 1) * CHUNK
            elif mode == "random":
                due = due or (t in rc.random_steps)
            if due:
                self.set_cache(self.make_forecast())
                refreshed = True
        if mode == "gate_null" and t < self.null_until:
            return None, 0, refreshed, disc
        s, age = self.slice()
        return s, age, refreshed, disc

    # -- one control step
    def act(self, rc: RunConfig) -> float:
        env, t = self.env, self.t
        tau = self.tau()
        g_star = expert_grip(env, t)
        fs, age, refreshed, disc = (None, 0, False, 0.0)
        hidden = None
        if rc.agent == "P":
            fs, age, refreshed, disc = self.update_forecast(rc)
            x = policy_input(env, self.tac, self.prop, t, tau, fs, age, True)
            if self.m.policy_p is None:   # DAgger iteration 0: teacher acts, student features recorded
                g = g_star
            else:
                y, hidden = self.m.policy_p(x[None], return_hidden=True)
                g = float(y[0, 0])
            if rc.record_features:
                xr = x
                if rc.p_null_record > 0 and self.rng.uniform() < rc.p_null_record:
                    xr = policy_input(env, self.tac, self.prop, t, tau, None, 0, True)
                self.log["x"].append(xr)
            if rc.action_probe and rc.probe_window[0] <= t < rc.probe_window[1]:
                self.log["probe"].append(self.probe(x, fs, age, g, g_star))
        elif rc.agent == "D":
            x = policy_input(env, self.tac, self.prop, t, tau, None, 0, False)
            if self.m.policy_d is None:
                g = g_star
            else:
                y, hidden = self.m.policy_d(x[None], return_hidden=True)
                g = float(y[0, 0])
            if rc.record_features:
                self.log["x"].append(x)
        elif rc.agent == "R":
            g = self.reactive(t, tau)
        elif rc.agent == "X":
            g = g_star
        else:
            raise ValueError(rc.agent)
        if rc.record_features and rc.agent in ("P", "D"):
            self.log["y"].append(g_star)
        if rc.record_hidden and hidden is not None:
            self.log["hidden"].append(hidden[0].copy())
        if rc.beta > 0 and self.rng.uniform() < rc.beta:
            g = g_star
        if rc.exec_noise > 0:
            g = g + self.rng.normal(0, rc.exec_noise)
        g = float(np.clip(g, 0.0, 1.0))
        self.log["g"].append(g)
        self.log["g_star"].append(g_star)
        self.log["refresh"].append(refreshed)
        self.log["disc"].append(disc)
        self.log["pred_now"].append(None if fs is None else fs[self.L - 1 if self.L > 0 else 0].copy())
        return g

    def reactive(self, t, tau) -> float:
        """Classic grip-force regulation from measured touch (no forecast, no learning):
        squeeze so that measured tangential / normal stays at a target slip margin."""
        rp: ReactiveParams = getattr(self, "reactive_params", ReactiveParams())
        if t < self.env.plan.close_start:
            return 0.0
        y = self.tac[tau]
        tang = max(y[2], y[3])
        n_target = max(rp.prior_gain * self.env.n_static_est, tang / rp.ratio)
        n_target = min(n_target, CRUSH_SAFETY * self.env.crush_limit)
        return grip_for_normal(n_target)

    def probe(self, x, fs, age, g, g_star) -> dict:
        """Paired action-level intervention: same state, live touch and context; only
        the forecast representation differs."""
        env, t, tau = self.env, self.t, self.tau()
        out = {"t": t, "g_cached": g, "g_star": g_star}
        refreshed = self.tac[tau] + self.m.forecaster(forecaster_input(
            self.tac, self.prop, np.asarray(self.grip_cmds), env.lift_cmd, tau, t, env.grip_applied)).reshape(HF, TAC)
        variants = {"refresh": (refreshed[:HP], 0), "null": (None, 0), "oracle": (self.oracle_forecast()[:HP], 0)}
        self.forecaster_calls -= 1  # probing is analysis, not deployment compute
        pool = getattr(self, "shuffle_pool", None)
        if pool is not None and len(pool):
            variants["shuffle"] = (pool[self.rng.integers(len(pool))], age)
        hid = {}
        for k, (s, a) in variants.items():
            xv = policy_input(env, self.tac, self.prop, t, tau, s, a, True)
            y, h = self.m.policy_p(xv[None], return_hidden=True)
            out["g_" + k] = float(y[0, 0])
            hid[k] = h[0]
        _, h0 = self.m.policy_p(x[None], return_hidden=True)
        hid["cached"] = h0[0]
        out["hidden"] = hid
        out["fc_err_cached"] = None if fs is None else float(np.abs(fs[max(self.L - 1, 0)] - self.tac_clean[t]).mean())
        out["fc_err_refresh"] = float(np.abs(refreshed[self.L - 1 if self.L > 0 else 0] - self.tac_clean[t]).mean())
        return out

    def step(self, rc: RunConfig):
        g = self.act(rc)
        self.grip_cmds.append(g)
        obs = self.env.step_grip(g)
        self.tac.append(obs["tactile"].copy())
        self.tac_clean.append(obs["tactile_clean"].copy())
        self.prop.append(obs["proprio"].copy())
        self.rel.append(self.env.box_rel_z())
        n = obs["tactile_clean"][:2]
        self.crush_run = self.crush_run + 1 if self.env.crushed(n) else 0
        self.crushed |= self.crush_run >= 2
        self.log["n_min"].append(float(n.min()))
        self.last_obs = obs
        self.t += 1

    def run_until(self, rc: RunConfig, t_end: int):
        while self.t < t_end:
            self.step(rc)

    # -- outcomes
    def fc_err_window(self, start: int, n: int = 10) -> float:
        errs = [float(np.abs(self.log["pred_now"][u] - self.tac_clean[u]).mean())
                for u in range(start, min(start + n, len(self.log["pred_now"])))
                if self.log["pred_now"][u] is not None]
        return float(np.mean(errs)) if errs else float("nan")

    def outcome(self, ref_step: int) -> dict:
        env, h2 = self.env, self.env.h2
        rel = np.asarray(self.rel)
        ref = rel[max(ref_step - 1, 0)]
        slip = float(ref - rel[-1])
        lifted = env.held(self.last_obs, h2.min_lift_frac * env.plan.lift_height)
        success = bool(lifted and not self.crushed and slip < h2.slip_tol)
        # recovery time: first step u >= ref after which slip speed < 5 mm/s for 5 steps
        rt = float("nan")
        if success:
            v = np.abs(np.diff(rel)) / env.cfg.control_dt
            for u in range(ref_step, len(v) - 4):
                if np.all(v[u:u + 5] < 0.005):
                    rt = (u - ref_step) * env.cfg.control_dt
                    break
        # forecast error of the forecast actually in use, for the 10 steps after ref
        fc_err = self.fc_err_window(ref_step)
        n_hold = np.asarray(self.log["n_min"][ref_step:])
        return {"success": success, "lifted": bool(lifted), "crushed": bool(self.crushed), "slip": slip,
                "recovery_time": rt, "fc_err_post": fc_err,
                "squeeze_ratio": float(np.mean(n_hold) / env.domain["n_static"]) if len(n_hold) else float("nan"),
                "refreshes_post": int(np.sum(self.log["refresh"][ref_step:])),
                "refreshes_total": int(np.sum(self.log["refresh"])),
                "forecaster_calls": self.forecaster_calls,
                "triggers_post": int(np.sum(np.asarray(self.log["disc"][ref_step:]) > getattr(self, "theta_log", np.inf))),
                "grip_mean_post": float(np.mean(self.log["g"][ref_step:])),
                "grip_err_post": float(np.mean(np.abs(np.asarray(self.log["g"][ref_step:]) -
                                                      np.asarray(self.log["g_star"][ref_step:]))))}
