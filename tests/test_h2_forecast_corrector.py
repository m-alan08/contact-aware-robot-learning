"""Tests for the H2-001 forecast/corrector heads (CPU, synthetic data only)."""
import torch

from contact_learning.h2.forecast_corrector import (
    INTERVENTIONS, ActivationStats, TactileCorrector, TactileForecaster, action_change,
    apply_intervention, corrector_target, masked_mse,
)

B, T, DA, DC = 6, 8, 6, 32
TAC = (3, 10, 14)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, DC, generator=g)
    live = torch.randn(B, *TAC, generator=g)
    acts = torch.randn(B, T, DA, generator=g)
    off = torch.randint(0, T, (B,), generator=g)
    fc = torch.randn(B, T, *TAC, generator=g)
    return ctx, live, acts, off, fc


def _corrector(**kw):
    torch.manual_seed(0)
    c = TactileCorrector(DC, DA, T, TAC, d_model=64, n_layers=2, n_heads=4, **kw)
    torch.nn.init.normal_(c.head.weight, std=0.1)  # non-zero head so outputs depend on inputs
    return c.eval()


def test_forecaster_shapes_and_residual():
    f = TactileForecaster(DC, DA, T, TAC, n_obs_steps=2, hidden=64)
    hist = torch.randn(B, 2, *TAC)
    out = f(torch.randn(B, DC), hist, torch.randn(B, T, DA))
    assert out.shape == (B, T, *TAC)
    torch.nn.init.zeros_(f.net[-1].weight); torch.nn.init.zeros_(f.net[-1].bias)
    out0 = f(torch.randn(B, DC), hist, torch.randn(B, T, DA))
    assert torch.allclose(out0, hist[:, -1:].expand_as(out0))  # zero residual = persistence


def test_initial_head_is_identity_correction():
    c = TactileCorrector(DC, DA, T, TAC, d_model=64, n_layers=2)
    ctx, live, acts, off, fc = _inputs()
    assert torch.count_nonzero(c(ctx, live, acts, off, fc)) == 0


def test_executed_prefix_never_corrected():
    c = _corrector()
    ctx, live, acts, off, fc = _inputs()
    d = c(ctx, live, acts, off, fc)
    for i in range(B):
        assert torch.all(d[i, : off[i]] == 0)
        if off[i] < T:
            assert torch.any(d[i, off[i]:] != 0)


def test_forecast_pathway_changes_action_with_other_inputs_fixed():
    c = _corrector()
    ctx, live, acts, off, fc = _inputs()
    d_cached = c(ctx, live, acts, off, fc)
    d_other = c(ctx, live, acts, off, fc + 1.0)
    d_null = c(ctx, live, acts, off, None)
    assert action_change(d_cached, d_other, off).min() > 0
    assert action_change(d_cached, d_null, off).min() > 0
    # same forecast, same everything -> identical (deterministic)
    assert torch.equal(d_cached, c(ctx, live, acts, off, fc))


def test_no_forecast_variant_ignores_forecast():
    c = _corrector(use_forecast=False)
    ctx, live, acts, off, fc = _inputs()
    assert torch.equal(c(ctx, live, acts, off, fc), c(ctx, live, acts, off, None))
    assert not hasattr(c, "forecast_proj")


def test_null_mask_equals_null_forecast_rowwise():
    c = _corrector()
    ctx, live, acts, off, fc = _inputs()
    mask = torch.tensor([True, False, True, False, False, True])
    d_mask = c(ctx, live, acts, off, fc, null_mask=mask)
    d_null = c(ctx, live, acts, off, None)
    d_full = c(ctx, live, acts, off, fc)
    assert torch.allclose(d_mask[mask], d_null[mask], atol=1e-6)
    assert torch.allclose(d_mask[~mask], d_full[~mask], atol=1e-6)


def test_scale_one_is_identity_and_zero_differs_from_null():
    c = _corrector()
    ctx, live, acts, off, fc = _inputs()
    assert torch.allclose(c(ctx, live, acts, off, fc, forecast_scale=1.0), c(ctx, live, acts, off, fc))
    # scale 0 removes forecast content AND the (trained, non-zero) null token: a distinct,
    # possibly OOD input. The null token is zero-initialised, so give it a trained-like value.
    with torch.no_grad():
        c.null_forecast.normal_()
    assert not torch.allclose(c(ctx, live, acts, off, fc, forecast_scale=0.0), c(ctx, live, acts, off, None))


def test_interventions_semantics():
    g = torch.Generator().manual_seed(1)
    cached, refreshed, other, oracle = (torch.randn(B, T, *TAC, generator=g) for _ in range(4))
    assert set(INTERVENTIONS) >= {"cached", "refresh", "null", "shuffle", "norm_matched", "oracle"}
    assert apply_intervention("cached", cached) is cached
    assert apply_intervention("refresh", cached, refreshed=refreshed) is refreshed
    assert apply_intervention("null", cached) is None
    assert apply_intervention("shuffle", cached, other_trial=other) is other
    assert apply_intervention("oracle", cached, oracle=oracle) is oracle
    nm = apply_intervention("norm_matched", cached, refreshed=refreshed, generator=g)
    want = (refreshed - cached).flatten(1).norm(dim=1)
    got = (nm - cached).flatten(1).norm(dim=1)
    assert torch.allclose(want, got, rtol=1e-5)
    assert not torch.allclose(nm, refreshed)


def test_corrector_target_and_loss_mask():
    plan, demo = torch.zeros(2, T, DA), torch.ones(2, T, DA)
    off = torch.tensor([0, 5])
    tgt, mask = corrector_target(plan, demo, off)
    assert mask[0].all() and mask[1, :5].sum() == 0 and mask[1, 5:].all()
    assert torch.all(tgt[1, :5] == 0) and torch.all(tgt[1, 5:] == 1)
    assert torch.isclose(masked_mse(tgt, tgt, mask), torch.tensor(0.0))


def test_activation_stats_flags_shifted_states():
    g = torch.Generator().manual_seed(2)
    nominal = torch.randn(2000, 16, generator=g)
    st = ActivationStats().fit(nominal)
    d_in = st.distance(torch.randn(500, 16, generator=g)).median()
    d_out = st.distance(torch.randn(500, 16, generator=g) + 3.0).median()
    assert d_out > 2 * d_in


def test_toy_learning_uses_forecast_and_live_disagreement():
    """Correct action = W (live - forecast[k]). A forecast-using corrector must beat the
    no-forecast variant, and swapping in another trial's forecast must hurt."""
    torch.manual_seed(0)
    tac, dc, da, t, n = (1, 2, 2), 4, 2, 4, 512
    W = torch.randn(4, da)

    def batch(gen):
        ctx = torch.randn(n, dc, generator=gen)
        fc = torch.randn(n, t, *tac, generator=gen)
        off = torch.randint(0, t, (n,), generator=gen)
        live = fc[torch.arange(n), off] + 0.8 * torch.randn(n, *tac, generator=gen)
        plan = torch.zeros(n, t, da)
        demo = ((live - fc[torch.arange(n), off]).reshape(n, -1) @ W)[:, None].expand(n, t, da)
        tgt, mask = corrector_target(plan, demo, off)
        return ctx, live, plan, off, fc, tgt, mask

    def train(use_forecast):
        torch.manual_seed(0)
        c = TactileCorrector(dc, da, t, tac, d_model=64, n_layers=2, n_heads=4,
                             use_forecast=use_forecast, p_null=0.0)
        opt = torch.optim.Adam(c.parameters(), lr=2e-3)
        gen = torch.Generator().manual_seed(3)
        for _ in range(250):
            ctx, live, plan, off, fc, tgt, mask = batch(gen)
            loss = masked_mse(c(ctx, live, plan, off, fc), tgt, mask)
            opt.zero_grad(); loss.backward(); opt.step()
        return c.eval()

    gen = torch.Generator().manual_seed(99)
    ctx, live, plan, off, fc, tgt, mask = batch(gen)
    with_fc, without_fc = train(True), train(False)
    with torch.no_grad():
        l_with = masked_mse(with_fc(ctx, live, plan, off, fc), tgt, mask).item()
        l_without = masked_mse(without_fc(ctx, live, plan, off, fc), tgt, mask).item()
        l_shuffled = masked_mse(with_fc(ctx, live, plan, off, fc.roll(1, 0)), tgt, mask).item()
    assert l_with < 0.5 * l_without
    assert l_shuffled > 2 * l_with
