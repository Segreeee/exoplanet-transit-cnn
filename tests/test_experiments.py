"""Esperimenti CNN di Fase 3: varianti, esecuzione appaiata alla baseline, riepilogo."""

import numpy as np
import pytest

pytest.importorskip("torch")

from exotransit.experiments import VARIANTS, run_experiments, summarize_runs  # noqa: E402


def _fake_arrays(n_groups=60, per_group=2, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    y = (rng.random(groups.size) < 0.4).astype(int)
    g = rng.normal(0, 0.1, (groups.size, 2001)).astype(np.float32)
    l_ = rng.normal(0, 0.1, (groups.size, 201)).astype(np.float32)
    l_[y == 1, 90:110] -= 1.0  # il transito e' nella vista locale
    extra = rng.normal(size=(groups.size, 4)).astype(np.float32)
    return g, l_, extra, y, groups


def test_variants_are_the_four_ablations_of_the_roadmap():
    assert set(VARIANTS) == {"full", "global_only", "local_only", "no_scalar"}


def test_run_experiments_records_every_variant_seed_and_baseline():
    out = run_experiments(*_fake_arrays(), seeds=[0, 1], variants=["full", "no_scalar"], epochs=2)
    assert sorted((r["variant"], r["seed"]) for r in out["runs"]) == [
        ("full", 0),
        ("full", 1),
        ("no_scalar", 0),
        ("no_scalar", 1),
    ]
    r = out["runs"][0]
    assert len(r["history"]) >= 1 and r["n_params"] > 0 and 0 <= r["test"]["roc_auc"] <= 1
    assert set(out["baseline_random_forest_by_seed"]) == {0, 1}
    # split per stella: il test di un seed non condivide stelle col suo train (garantito da
    # split_by_group); qui controlliamo che le dimensioni del test siano registrate
    assert r["test"]["n"] > 0


def test_summarize_runs_mean_std_and_paired_difference():
    def run(variant, seed, auc):
        return {
            "variant": variant,
            "seed": seed,
            "test": {"roc_auc": auc, "pr_auc": auc, "precision_at_recall90": auc},
        }

    runs = [run("full", s, a) for s, a in [(0, 0.90), (1, 0.92), (2, 0.94)]]
    rf = {s: {"roc_auc": 0.88, "pr_auc": 0.88, "precision_at_recall90": 0.88} for s in (0, 1, 2)}
    s = summarize_runs(runs, rf)
    m = s["full"]["metrics"]["roc_auc"]
    assert m["mean"] == pytest.approx(0.92) and m["std"] == pytest.approx(0.02)
    assert m["values"] == [0.90, 0.92, 0.94]
    d = s["full"]["paired_diff_vs_rf"]["roc_auc"]
    assert d["mean"] == pytest.approx(0.04) and d["n_seeds_better"] == 3
    assert s["full"]["n_seeds"] == 3


def test_shuffle_labels_by_group_keeps_groups_whole_and_class_counts():
    from exotransit.experiments import shuffle_labels_by_group

    g = np.repeat(np.arange(40), 2)
    y = np.repeat(np.arange(40) % 3 == 0, 2).astype(int)
    ys = shuffle_labels_by_group(y, g, seed=0)
    assert ys.sum() == y.sum() and (ys != y).any()
    assert all(len(set(ys[g == k])) == 1 for k in range(40))  # stessa etichetta per stella
    assert (ys == shuffle_labels_by_group(y, g, seed=0)).all()
