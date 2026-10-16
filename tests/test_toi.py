"""Tabella TOI di TESS -> stesse colonne della tabella KOI (nessuna rete)."""

import numpy as np
import pandas as pd
import pytest

from exotransit.koi import views_for_koi
from exotransit.toi import TOI_BTJD_OFFSET, prepare_toi_table


def _table():
    return pd.DataFrame(
        {
            "tid": [10, 11, 12, 13, 14, 15],
            "toi": ["1.01", "2.01", "3.01", "4.01", "5.01", "6.01"],
            "tfopwg_disp": ["CP", "FP", "KP", "FA", "PC", "CP"],
            "pl_orbper": [3.0, 5.0, 7.0, 9.0, 11.0, np.nan],
            "pl_tranmid": [2458500.0, 2458501.0, 2458502.0, 2458503.0, 2458504.0, 2458505.0],
            "pl_trandurh": [2.4, 3.6, 4.8, 6.0, 7.2, 1.0],
            "pl_trandep": [1000.0, 2000.0, 500.0, 300.0, 100.0, 400.0],
        }
    )


def test_labels_units_and_time_system():
    t = prepare_toi_table(_table())
    assert list(t["name"]) == ["TOI-1.01", "TOI-2.01", "TOI-3.01", "TOI-4.01"]
    assert list(t["label"]) == [
        1,
        0,
        1,
        0,
    ]  # CP/KP positivi, FP/FA negativi; PC e senza periodo fuori
    assert (
        t.loc[0, "kepid"] == 10
    )  # la colonna `kepid` contiene il TIC (stessa interfaccia della tabella KOI)
    assert t.loc[0, "t0"] == pytest.approx(2458500.0 - TOI_BTJD_OFFSET)  # BJD -> BTJD
    assert t.loc[0, "duration"] == pytest.approx(0.1)  # 2.4 h -> giorni
    assert t.loc[0, "depth"] == pytest.approx(1e-3)  # ppm -> frazione
    assert (t["snr"] == 0.0).all()  # i TOI non hanno SNR: segnaposto esplicito


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="mancanti"):
        prepare_toi_table(_table().drop(columns=["pl_trandep"]))


def test_views_for_koi_snr_is_optional_for_tess():
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 100.0, 0.02)
    flux = 1.0 + rng.normal(0, 1e-4, time.size)
    flux[np.abs((time % 5.0) - 2.5) < 0.1] -= 0.01
    row = pd.Series(
        {
            "kepid": 1,
            "name": "x",
            "period": 5.0,
            "t0": 2.5,
            "duration": 0.2,
            "depth": 0.01,
            "snr": np.nan,
            "label": 1,
        }
    )
    from exotransit.koi import SkipKOI

    with pytest.raises(SkipKOI, match="snr_mancante"):
        views_for_koi(time, flux, row, siblings=[])  # comportamento Kepler invariato
    v = views_for_koi(time, flux, row, siblings=[], require_snr=False)
    assert np.isfinite(v.extra).all()  # SNR assente -> 0, non NaN nelle feature


def test_include_candidates_keeps_pc_for_sibling_masking():
    t = prepare_toi_table(_table(), include_candidates=True)
    assert "TOI-5.01" in set(t["name"])  # PC: serve per mascherare i suoi transiti
    assert t.loc[t["name"] == "TOI-5.01", "label"].item() == 1
