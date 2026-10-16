"""Calibrazione, soglia per recall obiettivo e tassi per fascia (Fase 4)."""

import numpy as np
import pandas as pd
import pytest

from exotransit.errors import binned_rates
from exotransit.metrics import (
    expected_calibration_error,
    reliability_curve,
    threshold_for_recall,
)


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)  # etichette generate dalle probabilita' stesse
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.where(p > 0.5, 0.99, 0.01)) > 0.2


def test_reliability_curve_bins_and_counts():
    p = np.array([0.05, 0.05, 0.95, 0.95, 0.95])
    y = np.array([0, 1, 1, 1, 0])
    mean_p, frac, count = reliability_curve(y, p, n_bins=10)
    assert count.tolist() == [2, 3] and frac.tolist() == pytest.approx([0.5, 2 / 3])
    assert mean_p.tolist() == pytest.approx([0.05, 0.95])
    assert count.sum() == len(y)  # il valore 1.0 finisce nell'ultimo bin, nessun punto perso
    assert reliability_curve(np.array([1]), np.array([1.0]), 10)[2].tolist() == [1]


def test_threshold_for_recall_is_highest_threshold_reaching_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.5, 0.2, 0.1])
    assert threshold_for_recall(y, s, 0.75) == pytest.approx(0.7)  # 3/4 positivi
    assert threshold_for_recall(y, s, 1.0) == pytest.approx(0.3)
    assert np.isnan(threshold_for_recall(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.9))


def test_binned_rates_counts_recall_precision_fpr():
    df = pd.DataFrame(
        {
            "period": [1, 1, 1, 1, 10, 10, 10, 10],
            "label": [1, 1, 0, 0, 1, 1, 0, 0],
            "pred": [True, False, True, False, True, True, False, False],
        }
    )
    t = binned_rates(df, "period", [0, 5, 100]).set_index("bin")
    low, high = t.iloc[0], t.iloc[1]
    assert (low["n"], low["n_pos"]) == (4, 2)
    assert low["recall"] == 0.5 and low["precision"] == 0.5 and low["fpr"] == 0.5
    assert high["recall"] == 1.0 and high["precision"] == 1.0 and high["fpr"] == 0.0


def test_binned_rates_empty_denominators_are_nan_not_errors():
    df = pd.DataFrame({"x": [1, 1], "label": [0, 0], "pred": [False, False]})
    t = binned_rates(df, "x", [0, 2])
    assert (
        t["n"].tolist() == [2]
        and np.isnan(t["recall"].iloc[0])
        and np.isnan(t["precision"].iloc[0])
    )
