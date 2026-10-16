"""Trasferimento Kepler -> TESS (Fase 5) su dati finti minuscoli."""

import numpy as np
import pytest

pytest.importorskip("torch")

from exotransit.experiments import run_experiments  # noqa: E402
from exotransit.transfer import APPROACHES, run_transfer, summarize_transfer  # noqa: E402


def _fake(n_groups=60, per_group=2, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    y = (rng.random(groups.size) < 0.45).astype(int)
    g = rng.normal(0, 0.1, (groups.size, 2001)).astype(np.float32)
    l_ = rng.normal(0, 0.1, (groups.size, 201)).astype(np.float32)
    l_[y == 1, 90:110] -= 1.0
    extra = rng.normal(size=(groups.size, 4)).astype(np.float32)
    extra[:, 3] = 0.0  # TESS: nessun SNR
    return g, l_, extra, y, groups


def test_run_transfer_covers_all_approaches_and_keeps_stars_disjoint(tmp_path):
    kep = _fake(seed=1)
    run_experiments(*kep, seeds=[0], variants=["no_scalar"], epochs=2, models_dir=tmp_path)
    tess = _fake(seed=2)
    out = run_transfer(
        *tess, models_dir=tmp_path, seeds=[0], epochs_finetune=2, epochs_scratch=2, n_boot=20
    )
    r = out["runs"][0]
    assert set(r["approaches"]) == set(APPROACHES)
    sp = r["split"]
    assert sp["n_train"] + sp["n_val"] + sp["n_test"] == len(tess[3])
    for a in APPROACHES:
        m = r["approaches"][a]["test"]
        assert 0.0 <= m["roc_auc"] <= 1.0 and m["n"] == sp["n_test"]
    whole = out["kepler_as_is_all_tess"][0]["test"]
    assert whole["n"] == len(tess[3]) and len(whole["ci95"]["pr_auc"]) == 2


def test_summarize_transfer_mean_std_and_paired_difference():
    def run(seed, a, b, c, d):
        mk = lambda v: {"test": {"roc_auc": v, "pr_auc": v, "precision_at_recall90": v}}  # noqa: E731
        return {
            "seed": seed,
            "approaches": {
                "kepler_as_is": mk(a),
                "fine_tuned": mk(b),
                "tess_from_scratch": mk(c),
                "random_forest_3_features": mk(d),
            },
        }

    runs = [run(0, 0.60, 0.80, 0.70, 0.65), run(1, 0.62, 0.84, 0.72, 0.67)]
    s = summarize_transfer(runs)
    assert s["fine_tuned"]["metrics"]["pr_auc"]["mean"] == pytest.approx(0.82)
    d = s["fine_tuned"]["paired_diff"]["vs_kepler_as_is"]["pr_auc"]
    assert d["mean"] == pytest.approx(0.21) and d["n_seeds_better"] == 2
    assert s["fine_tuned"]["paired_diff"]["vs_tess_from_scratch"]["pr_auc"][
        "mean"
    ] == pytest.approx(0.11)
    assert s["fine_tuned"]["paired_diff"]["vs_random_forest_3_features"]["pr_auc"][
        "mean"
    ] == pytest.approx(0.16)
