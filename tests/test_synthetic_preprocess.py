import numpy as np
import pytest

from exotransit.preprocess import (
    GLOBAL_BINS,
    LOCAL_BINS,
    N_EXTRA,
    Ephemeris,
    bin_view,
    detrend,
    normalize_view,
    process_light_curve,
    running_median,
    transit_mask,
    wrapped_phase,
)
from exotransit.synthetic import KINDS, make_dataset, make_example, trapezoid_profile


def test_wrapped_phase_range_and_center():
    t = np.linspace(0, 100, 5000)
    ph = wrapped_phase(t, period=7.0, t0=2.0)
    assert ph.min() >= -3.5 and ph.max() < 3.5
    assert abs(wrapped_phase(np.array([2.0 + 7.0 * 5]), 7.0, 2.0)[0]) < 1e-9


def test_trapezoid_profile_shape():
    ph = np.linspace(-0.5, 0.5, 1001)
    prof = trapezoid_profile(ph, duration=0.2, ingress_frac=0.2)
    assert prof[500] == pytest.approx(1.0)  # centro: profondita' piena
    assert prof[0] == 0.0 and prof[-1] == 0.0  # fuori dal transito
    assert 0.0 < prof[int(500 + 0.095 * 1000)] < 1.0  # sull'ingresso
    v = trapezoid_profile(ph, duration=0.2, ingress_frac=0.5)  # forma a V
    assert v[500] == pytest.approx(1.0) and v[500 + 50] == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        trapezoid_profile(ph, 0.2, ingress_frac=0.0)


def test_make_example_is_deterministic_and_labelled():
    a = make_example(np.random.default_rng(3), kind="planet")
    b = make_example(np.random.default_rng(3), kind="planet")
    np.testing.assert_array_equal(a.flux, b.flux)
    assert a.label == 1 and np.isfinite(a.period) and a.depth > 0
    for kind in ("eb", "noise"):
        ex = make_example(np.random.default_rng(4), kind=kind)
        assert ex.label == 0
    assert np.isnan(make_example(np.random.default_rng(4), kind="noise").period)
    with pytest.raises(ValueError):
        make_example(np.random.default_rng(0), kind="boh")


def test_make_dataset_groups_and_kinds():
    ds = make_dataset(40, seed=1, examples_per_star=2)
    assert len(ds) == 40
    assert {e.kind for e in ds} <= set(KINDS)
    assert ds[0].star_id == ds[1].star_id != ds[2].star_id


def test_running_median_removes_slow_trend_but_keeps_short_dip():
    t = np.arange(0, 30, 0.02)
    slow = 0.01 * np.sin(2 * np.pi * t / 12.0)
    flux = 1.0 + slow
    flux[(t > 10) & (t < 10.1)] -= 0.005  # dip breve
    flat = detrend(t, flux, window_days=1.5)
    assert np.std(flat[t < 8]) < 5e-4  # il trend lento e' sparito
    assert flat[(t > 10.02) & (t < 10.08)].mean() < 1 - 0.003  # il dip sopravvive


def test_masked_detrend_preserves_transit_depth_better():
    t = np.arange(0, 20, 0.02)
    flux = np.ones_like(t)
    inside = np.abs(wrapped_phase(t, 5.0, 2.5)) < 0.3  # transito lungo (0.6 d) e profondo
    flux[inside] -= 0.01
    unmasked = detrend(t, flux, window_days=1.0)
    masked = detrend(t, flux, window_days=1.0, mask=transit_mask(t, 5.0, 2.5, 0.6))
    assert (1 - masked[inside].mean()) > (1 - unmasked[inside].mean())
    assert (1 - masked[inside].mean()) > 0.0095


def test_running_median_needs_enough_points():
    with pytest.raises(ValueError):
        running_median(np.arange(5.0), np.ones(5), 1.0)


def test_bin_view_interpolates_empty_bins_and_handles_empty_input():
    x = np.array([0.1, 0.12, 0.9])
    y = np.array([1.0, 3.0, 5.0])
    out = bin_view(x, y, 0.0, 1.0, 10)
    assert out.shape == (10,) and np.isfinite(out).all()
    assert out[1] == pytest.approx(2.0)  # mediana di 1 e 3
    assert out[9] == pytest.approx(5.0)
    assert np.all(bin_view(np.array([5.0]), np.array([1.0]), 0.0, 1.0, 4) == 0.0)


def test_normalize_view_median_zero_min_minus_one():
    v = normalize_view(np.array([1.0, 1.0, 0.9, 1.0, 1.0]))
    assert np.median(v) == 0.0 and v.min() == pytest.approx(-1.0)
    assert np.all(normalize_view(np.ones(5)) == 0.0)  # vista piatta: nessuna divisione per zero


def test_pipeline_recovers_period_and_builds_views_for_planets():
    rng = np.random.default_rng(1)
    recovered = 0
    for i in range(6):
        ex = make_example(rng, kind="planet", star_id=i)
        views, eph = process_light_curve(ex.time, ex.flux, max_periods=6000)
        recovered += abs(eph.period / ex.period - 1) < 0.02
        assert views.global_view.shape == (GLOBAL_BINS,)
        assert views.local_view.shape == (LOCAL_BINS,)
        assert views.extra.shape == (N_EXTRA,) and np.isfinite(views.extra).all()
        assert views.local_view.min() == pytest.approx(-1.0)
    assert recovered >= 5


def test_pipeline_with_known_ephemeris_skips_bls():
    ex = make_example(np.random.default_rng(7), kind="planet")
    eph = Ephemeris(ex.period, ex.t0, ex.duration, ex.depth, snr=20.0)
    views, used = process_light_curve(ex.time, ex.flux, ephemeris=eph)
    assert used is eph
    center = LOCAL_BINS // 2
    # il minimo della vista locale e' vicino al centro: il transito e' allineato
    assert abs(int(np.argmin(views.local_view)) - center) < 0.15 * LOCAL_BINS


def test_pipeline_ignores_nan_and_unsorted_input():
    ex = make_example(np.random.default_rng(2), kind="planet")
    order = np.random.default_rng(0).permutation(ex.time.size)
    flux = ex.flux.copy()
    flux[5] = np.nan
    views, _ = process_light_curve(ex.time[order], flux[order], max_periods=3000)
    assert np.isfinite(views.global_view).all()


def test_ephemeris_from_extra_roundtrip():
    from exotransit.preprocess import Ephemeris, ephemeris_from_extra, make_views

    rng = np.random.default_rng(0)
    time = np.arange(0.0, 60.0, 0.02)
    flux = 1.0 + rng.normal(0.0, 1e-4, time.size)
    eph = Ephemeris(period=7.3, t0=1.0, duration=0.15, depth=2e-3, snr=25.0)
    back = ephemeris_from_extra(make_views(time, flux, eph).extra)
    assert back.period == pytest.approx(eph.period, rel=1e-5)
    assert back.duration == pytest.approx(eph.duration, rel=1e-5)
    assert back.depth == pytest.approx(eph.depth, rel=1e-5)
    assert back.snr == pytest.approx(eph.snr, rel=1e-5)
