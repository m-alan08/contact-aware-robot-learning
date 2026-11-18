"""H1-A paired trials: does correcting an action make the cached (original-action)
forecast less accurate than a refreshed (corrected-action) forecast?

One trial = one held-out episode seed = one correction. Procedure:
 1. reset(seed); sample a nominal plan from the trial seed.
 2. roll the nominal plan out once (reference) to find the nominal contact onset.
 3. choose the correction time t_c (stratified: near the nominal onset, or uniform).
 4. reset(seed) again and execute the nominal plan up to t_c; record history h.
 5. A_orig = plan[t_c:t_c+H]; A_corr = clip(A_orig + correction).
 6. snapshot state; execute A_orig -> Y_orig (counterfactual, for labelling only);
    restore; execute A_corr -> Y_corr (the trajectory both forecasts are scored on).
 7. cached = f(h, A_orig); refreshed = f(h, A_corr); sham = f(h, A_sham) where A_sham
    is a *different* correction of the same requested magnitude that is never executed.

Simulation is separated from prediction so that every predictor is scored on exactly
the same trials (``simulate_trials`` then ``score_trials``).
"""
from __future__ import annotations

import time

import numpy as np

from contact_learning.controllers.scripted import (CORRECTION_TYPES, Correction, apply_correction,
                                                   effective_magnitude, nominal_plan, sample_plan_params)
from contact_learning.environments.gripper_lift import GripperLiftEnv
from contact_learning.evaluation import metrics as M
from contact_learning.evaluation.rollout import Recorder, continue_rollout, rollout
from contact_learning.predictors.features import flatten_inputs, history_at


def grasp_contact(tactile_clean: np.ndarray, thr: float) -> np.ndarray:
    """Binary grasp-contact state: both pads above the normal-force threshold."""
    return (tactile_clean[..., 0] > thr) & (tactile_clean[..., 1] > thr)


def simulate_trial(env: GripperLiftEnv, seed: int, cfg: dict) -> dict:
    T, K, H = cfg["T"], cfg["K"], cfg["H"]
    thr = env.cfg.contact_threshold
    rng = np.random.default_rng([seed, 2])
    p = sample_plan_params(rng)
    plan = nominal_plan(p, T, env.cfg.max_lift_speed, env.cfg.control_dt)

    nom = rollout(env, seed, plan)
    c_nom = grasp_contact(nom.tactile_clean, thr)
    onset = int(np.argmax(c_nom)) if c_nom.any() else -1

    lo, hi = K, T - H
    stratum = str(rng.choice(cfg.get("strata", ["near_onset", "uniform"])))
    if stratum == "near_onset" and onset >= 0:
        cand = np.arange(max(lo, onset - H + 1), min(hi, onset) + 1)
        t_c = int(rng.choice(cand)) if len(cand) else int(rng.integers(lo, hi + 1))
    else:
        t_c = int(rng.integers(lo, hi + 1))

    kind = str(rng.choice(CORRECTION_TYPES))
    mag = float(rng.choice(cfg["magnitudes"]))
    sham_kind = str(rng.choice([k for k in CORRECTION_TYPES if k != kind]))

    # Re-execute nominal plan up to t_c (same seed => identical to the reference prefix).
    obs = env.reset(seed)
    rec = Recorder(obs)
    for t in range(t_c):
        rec.add(plan[t], env.step(plan[t]))
    pre = rec.trajectory()
    prefix_ok = bool(np.array_equal(pre.tactile, nom.tactile[:t_c + 1]))
    h = history_at(pre.tactile, pre.proprio, pre.actions, t_c, K)
    state = env.get_state()

    A_orig = plan[t_c:t_c + H].copy()
    A_corr = apply_correction(A_orig, Correction(kind, mag))
    A_sham = apply_correction(A_orig, Correction(sham_kind, mag))

    Y_orig = continue_rollout(env, A_orig)
    env.set_state(state)
    Y_corr = continue_rollout(env, A_corr)

    win = np.arange(max(0, t_c - 2), t_c + H + 1)
    near_transition = bool(len(np.unique(c_nom[win])) > 1)
    stable_contact = bool(c_nom[max(0, t_c - K + 1):t_c + H + 1].all())
    phase = "near_transition" if near_transition else ("stable_contact" if stable_contact else "no_contact")

    cs_orig = grasp_contact(Y_orig["tactile_clean"], thr)
    cs_corr = grasp_contact(Y_corr["tactile_clean"], thr)
    return {
        "seed": seed, "t_c": t_c, "stratum": stratum, "onset": onset, "kind": kind,
        "magnitude_requested": mag, "magnitude_effective": effective_magnitude(A_orig, A_corr),
        "sham_kind": sham_kind, "sham_magnitude_effective": effective_magnitude(A_orig, A_sham),
        "phase": phase, "near_transition": near_transition,
        "contact_changing": bool((cs_orig != cs_corr).any()),
        "transition_in_executed": bool(len(np.unique(np.concatenate([[grasp_contact(pre.tactile_clean[-1], thr)], cs_corr]))) > 1),
        "outcome_change_rmse": float(M.forecast_rmse(Y_corr["tactile"], Y_orig["tactile"])),
        "mass": env.domain["mass"], "friction": env.domain["friction"], "g_hold": p.g_hold,
        "prefix_reproducible": prefix_ok,
        # arrays
        "_X": flatten_inputs(h, None), "_hist_key": h.key(),
        "_A_orig": A_orig, "_A_corr": A_corr, "_A_sham": A_sham,
        "_Y_corr": Y_corr["tactile"], "_Y_corr_clean": Y_corr["tactile_clean"], "_Y_orig": Y_orig["tactile"],
    }


def simulate_trials(env: GripperLiftEnv, seeds: list[int], cfg: dict) -> list[dict]:
    return [simulate_trial(env, s, cfg) for s in seeds]


def stack(trials: list[dict], key: str) -> np.ndarray:
    return np.stack([t[key] for t in trials])


def score_trials(trials: list[dict], predictor, thr: float, latency_samples: int = 100) -> dict:
    """Returns per-trial metric arrays for one predictor."""
    X = stack(trials, "_X")
    Ao, Ac, As = (stack(trials, k).reshape(len(trials), -1) for k in ("_A_orig", "_A_corr", "_A_sham"))
    Y, Yc, Yo = stack(trials, "_Y_corr"), stack(trials, "_Y_corr_clean"), stack(trials, "_Y_orig")
    # Same X object is used for both forecasts: the only difference is the action input.
    f_c = predictor.predict(X, Ao)
    f_r = predictor.predict(X, Ac)
    f_s = predictor.predict(X, As)
    lat = []
    for i in range(min(latency_samples, len(trials))):
        t0 = time.perf_counter()
        predictor.predict(X[i:i + 1], Ac[i:i + 1])
        lat.append(time.perf_counter() - t0)
    out = {
        "E_cached": M.forecast_rmse(f_c, Y), "E_refreshed": M.forecast_rmse(f_r, Y), "E_sham": M.forecast_rmse(f_s, Y),
        "E_cached_clean": M.forecast_rmse(f_c, Yc), "E_refreshed_clean": M.forecast_rmse(f_r, Yc),
        "En_cached": M.normal_rmse(f_c, Y), "En_refreshed": M.normal_rmse(f_r, Y),
        "CSE_cached": M.contact_state_error(f_c, Yc, thr), "CSE_refreshed": M.contact_state_error(f_r, Yc, thr),
        "E_cached_on_orig": M.forecast_rmse(f_c, Yo),  # accuracy of cached forecast for the action it was made for
        "forecast_shift": M.forecast_rmse(f_r, f_c),   # how much the predictor's output responds to the correction
        "PH_cached": M.per_horizon_rmse(f_c, Y), "PH_refreshed": M.per_horizon_rmse(f_r, Y),
    }
    out["D"] = M.paired_difference(out["E_cached"], out["E_refreshed"])
    out["D_sham"] = M.paired_difference(out["E_cached"], out["E_sham"])
    out["latency_s"] = np.array(lat)
    return out
