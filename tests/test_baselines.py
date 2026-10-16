"""Intervalli bootstrap per gruppo e baseline di Fase 2 (dati sintetici minuscoli, nessuna rete)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from exotransit.baselines import baseline_report
from exotransit.metrics import bootstrap_ci


def _toy(n_groups=60, per_group=2, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    y = (rng.random(groups.size) < 0.4).astype(int)
    score = y + rng.normal(0.0, 0.8, y.size)
    return y, score, groups


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    y, s, g = _toy()
    point = roc_auc_score(y, s)
    lo, hi = bootstrap_ci(roc_auc_score, y, s, g, n_boot=300, seed=1)
    assert lo <= point <= hi and 0.0 < hi - lo < 0.5
    assert (lo, hi) == bootstrap_ci(roc_auc_score, y, s, g, n_boot=300, seed=1)


def test_bootstrap_ci_resamples_whole_groups():
    n_groups = 30
    g = np.repeat(np.arange(n_groups), 2)
    y = np.tile([0, 1], n_groups)
    s = np.arange(g.size, dtype=float)  # punteggio unico per riga: identifica la riga estratta
    calls = []

    def metric(yy, ss):
        ids, counts = np.unique(ss.astype(int), return_counts=True)
        c = dict(zip(ids, counts, strict=True))
        # le due righe di ogni stella (2k, 2k+1) compaiono sempre lo stesso numero di volte
        assert all(c.get(2 * k, 0) == c.get(2 * k + 1, 0) for k in range(n_groups))
        calls.append(1)
        return 0.5

    bootstrap_ci(metric, y, s, g, n_boot=40, seed=0)
    assert len(calls) == 40


def test_bootstrap_ci_perfect_separation_is_degenerate():
    y = np.array([0, 1] * 30)
    g = np.arange(60)
    lo, hi = bootstrap_ci(roc_auc_score, y, y.astype(float), g, n_boot=100, seed=0)
    assert lo == hi == 1.0


def test_baseline_report_structure_and_group_disjoint_splits():
    rng = np.random.default_rng(0)
    n_groups = 120
    groups = np.repeat(np.arange(n_groups), 2)
    y = (rng.random(groups.size) < 0.4).astype(int)
    extra = rng.normal(size=(groups.size, 4)).astype(np.float32)
    extra[:, 3] += 1.5 * y  # l'SNR separa le classi
    r = baseline_report(extra, y, groups, seed=0, n_boot=50)
    assert set(r["models"]) == {"snr_only", "random_forest_snr_only", "random_forest_scalar"}
    assert sum(s["n"] for s in r["splits"].values()) == y.size
    assert sum(s["n_stars"] for s in r["splits"].values()) == n_groups  # nessuna stella divisa
    m = r["models"]["snr_only"]["test"]
    assert m["n"] == r["splits"]["test"]["n"]
    lo, hi = m["ci95"]["roc_auc"]
    assert lo <= m["roc_auc"] <= hi
    assert m["roc_auc"] > 0.7
    assert 0.0 <= r["models"]["snr_only"]["threshold_from_val"] and "tp" in m["at_val_threshold"]
    with pytest.raises(ValueError):
        baseline_report(extra[:5], y[:5], groups[:5], seed=0, n_boot=10)


def test_non_monotonic_snr_is_captured_by_forest_but_not_by_threshold():
    rng = np.random.default_rng(1)
    n_groups = 400
    groups = np.repeat(np.arange(n_groups), 2)
    snr = rng.uniform(0.0, 3.0, groups.size)
    y = ((snr > 1.0) & (snr < 2.0)).astype(int)  # positivi solo nella fascia centrale
    extra = np.zeros((groups.size, 4), dtype=np.float32)
    extra[:, 3] = snr
    r = baseline_report(extra, y, groups, seed=0, n_boot=20)["models"]
    assert r["snr_only"]["test"]["roc_auc"] < 0.65
    assert r["random_forest_snr_only"]["test"]["roc_auc"] > 0.95
