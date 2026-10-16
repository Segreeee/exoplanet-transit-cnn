"""Iniezione di transiti e perturbazioni delle curve (Fase 4)."""

import numpy as np
import pytest

from exotransit.perturb import (
    inject_transit,
    perturb_light_curve,
    point_scatter,
    transit_duration_days,
)
from exotransit.preprocess import wrapped_phase


def _curve(n=20000, sigma=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.02
    return t, 1.0 + rng.normal(0, sigma, n)


def test_inject_transit_depth_and_out_of_transit_untouched():
    t, f = _curve()
    out = inject_transit(t, f, period=10.0, t0=3.0, depth=0.01, duration=0.3)
    phase = np.abs(wrapped_phase(t, 10.0, 3.0))
    assert np.mean(out[phase < 0.05] / f[phase < 0.05]) == pytest.approx(
        0.99, abs=1e-6
    )  # fondo piatto
    far = phase > 0.2
    np.testing.assert_array_equal(out[far], f[far])
    np.testing.assert_array_equal(inject_transit(t, f, 10.0, 3.0, 0.0, 0.3), f)


def test_transit_duration_scales_with_cube_root_of_period():
    assert transit_duration_days(365.25) == pytest.approx(13.0 / 24.0)
    assert transit_duration_days(8 * 365.25) / transit_duration_days(365.25) == pytest.approx(2.0)


def test_point_scatter_is_robust_and_matches_white_noise_sigma():
    t, f = _curve(sigma=2e-4)
    assert point_scatter(f) == pytest.approx(2e-4, rel=0.05)
    f2 = f.copy()
    f2[::500] -= 0.05  # outlier/transiti isolati non cambiano la stima
    assert point_scatter(f2) == pytest.approx(2e-4, rel=0.1)


def test_perturb_truncate_keeps_first_days_only():
    t, f = _curve()
    tt, ff = perturb_light_curve(t, f, "truncate", 100.0, np.random.default_rng(0))
    assert tt.max() < t[0] + 100.0 and tt.size == ff.size < t.size
    np.testing.assert_array_equal(ff, f[: ff.size])


def test_perturb_noise_scales_total_scatter_by_level_and_is_deterministic():
    t, f = _curve()
    # seed diverso da `_curve`: a seed uguale il rumore coincide e si somma linearmente
    _, f4 = perturb_light_curve(t, f, "noise", 4.0, np.random.default_rng(123))
    assert point_scatter(f4) == pytest.approx(4 * point_scatter(f), rel=0.1)
    _, f4b = perturb_light_curve(t, f, "noise", 4.0, np.random.default_rng(123))
    np.testing.assert_array_equal(f4, f4b)
    _, f1 = perturb_light_curve(t, f, "noise", 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(f1, f)
    with pytest.raises(ValueError):
        perturb_light_curve(t, f, "boh", 2.0, np.random.default_rng(0))
