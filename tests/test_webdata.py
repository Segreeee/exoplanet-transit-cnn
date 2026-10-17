"""Dati per la demo web: selezione degli esempi, finestra della curva, fatti calibrati."""

import numpy as np
import pytest

from exotransit.webdata import (
    bin_window,
    calibrated_facts,
    select_examples,
    transit_window,
)


def test_select_examples_balances_outcomes_and_is_deterministic():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.4).astype(int)
    scores = np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.3, n)
    idx = np.arange(n)
    pick = select_examples(
        y, scores, thr=0.5, idx=idx, n_each={"TP": 5, "TN": 5, "FP": 3, "FN": 3}, seed=1
    )
    assert [p["outcome"] for p in pick].count("TP") == 5
    assert [p["outcome"] for p in pick].count("FN") == 3
    assert len({p["index"] for p in pick}) == len(pick)  # nessun duplicato
    for p in pick:
        o = p["outcome"]
        assert (o in ("TP", "FN")) == (y[p["index"]] == 1)
        assert (o in ("TP", "FP")) == (scores[p["index"]] >= 0.5)
    again = select_examples(y, scores, 0.5, idx, {"TP": 5, "TN": 5, "FP": 3, "FN": 3}, seed=1)
    assert [p["index"] for p in pick] == [p["index"] for p in again]


def test_select_examples_takes_fewer_when_a_group_is_small():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])  # nessun errore
    pick = select_examples(
        y, scores, 0.5, np.arange(4), {"TP": 5, "TN": 5, "FP": 3, "FN": 3}, seed=0
    )
    assert len(pick) == 4


def test_transit_window_picks_dense_epoch_and_lists_transits():
    t = np.arange(0.0, 1000.0, 0.0204)
    t = t[~((t > 100) & (t < 400))]  # buco lungo: la finestra deve cadere dove ci sono dati
    start, end, transits = transit_window(t, period=10.0, t0=5.0, duration=0.2)
    assert end - start == pytest.approx(40.0)  # 4 periodi
    assert all(start <= x <= end for x in transits)
    assert len(transits) in (4, 5)  # centrata su un transito: i due agli estremi sono inclusi
    assert not (100 < (start + end) / 2 < 400)
    # periodo lungo: finestra limitata a max_days, almeno un transito dentro
    s2, e2, tr2 = transit_window(t, period=300.0, t0=5.0, duration=0.5, max_days=80.0)
    assert e2 - s2 == pytest.approx(80.0) and len(tr2) >= 1


def test_bin_window_limits_points_and_keeps_range():
    t = np.linspace(0, 40, 3000)
    f = 1 + 1e-4 * np.sin(t)
    out = bin_window(t, f, 0.0, 40.0, max_points=500)
    assert len(out["t"]) <= 500 and len(out["t"]) == len(out["f"])
    short = bin_window(t[:100], f[:100], 0.0, 2.0, max_points=500)
    assert len(short["t"]) == 100  # sotto il limite: nessuna perdita


def test_calibrated_facts_uses_raw_score_for_the_verdict():
    val_scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    val_y = np.array([0, 0, 0, 1, 1, 1])
    test_scores = np.array([0.05, 0.7])
    from exotransit.preprocess import ephemeris_from_extra

    extra = np.array([[-3.0, 0.8, 0.02, 3.0], [-3.0, 0.8, 0.02, 3.0]], dtype=np.float32)
    facts = calibrated_facts(
        names=["K1.01", "K2.01"],
        extra=extra,
        scores=test_scores,
        val_scores=val_scores,
        val_y=val_y,
        thr=0.5,
    )
    assert [f["esito_classificatore"] for f in facts] == ["scartato", "candidato"]
    assert facts[1]["probabilita_pianeta"] >= facts[0]["probabilita_pianeta"]
    assert facts[0]["candidato"] == "K1.01"
    assert facts[0]["periodo_giorni"] == pytest.approx(
        ephemeris_from_extra(extra[0]).period, abs=1e-3
    )
