import numpy as np
import pandas as pd
import pytest

from exotransit.koi import (
    SkipKOI,
    prepare_koi_table,
    row_to_ephemeris,
    sibling_ephemerides,
    views_for_koi,
)
from exotransit.preprocess import GLOBAL_BINS, LOCAL_BINS


def _table():
    return pd.DataFrame(
        {
            "kepid": [1, 2, 3, 4, 5],
            "kepoi_name": ["K00001.01", "K00002.01", "K00003.01", "K00004.01", "K00005.01"],
            "koi_disposition": [
                "CONFIRMED",
                "FALSE POSITIVE",
                "CANDIDATE",
                "CONFIRMED",
                "CONFIRMED",
            ],
            "koi_period": [10.0, 3.0, 5.0, np.nan, 7.0],
            "koi_time0bk": [100.0, 120.0, 130.0, 140.0, 150.0],
            "koi_duration": [2.4, 6.0, 3.0, 2.0, 0.0],  # ore
            "koi_depth": [1000.0, 50000.0, 800.0, 900.0, 700.0],  # ppm
            "koi_model_snr": [30.0, np.nan, 12.0, 9.0, 8.0],
        }
    )


def test_prepare_default_drops_candidates_and_invalid_rows():
    t = prepare_koi_table(_table())
    assert list(t["kepid"]) == [1, 2]  # 3 candidato, 4 senza periodo, 5 durata nulla
    assert list(t["label"]) == [1, 0]
    assert t.loc[0, "duration"] == pytest.approx(0.1)  # 2.4 h -> giorni
    assert t.loc[0, "depth"] == pytest.approx(1e-3)  # ppm -> frazione
    assert np.isnan(t.loc[1, "snr"])  # SNR mancante: resta NaN, non diventa 0


def test_prepare_with_candidates_marks_them_positive():
    t = prepare_koi_table(_table(), include_candidates=True)
    assert list(t["kepid"]) == [1, 2, 3]
    assert t.loc[t["kepid"] == 3, "label"].item() == 1


def test_prepare_rejects_missing_columns():
    with pytest.raises(ValueError, match="mancanti"):
        prepare_koi_table(_table().drop(columns=["koi_depth"]))


def test_row_to_ephemeris():
    row = prepare_koi_table(_table()).iloc[0]
    eph = row_to_ephemeris(row)
    assert (eph.period, eph.t0, eph.snr) == (10.0, 100.0, 30.0)


def _multi_table():
    return pd.DataFrame(
        {
            "kepid": [1, 1, 1, 2],
            "kepoi_name": ["K1.01", "K1.02", "K1.03", "K2.01"],
            "koi_disposition": ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE", "CONFIRMED"],
            "koi_period": [5.0, 3.1, 11.0, 7.0],
            "koi_time0bk": [2.5, 1.0, 4.0, 1.0],
            "koi_duration": [4.8, 4.8, 4.8, 4.8],
            "koi_depth": [10000.0, 10000.0, 10000.0, 10000.0],
            "koi_model_snr": [50.0, 40.0, 30.0, 20.0],
        }
    )


def test_siblings_include_all_dispositions_of_the_same_star_only():
    all_kois = prepare_koi_table(_multi_table(), include_candidates=True)
    sib = sibling_ephemerides(all_kois, kepid=1, name="K1.01")
    assert sorted(e.period for e in sib) == [3.1, 11.0]  # CANDIDATE e FALSE POSITIVE inclusi
    assert sibling_ephemerides(all_kois, kepid=2, name="K2.01") == []


def _curve(t_end=100.0):
    rng = np.random.default_rng(0)
    t = np.arange(0.0, t_end, 0.02)
    flux = 1.0 + rng.normal(0.0, 1e-4, t.size)
    flux[np.abs(((t - 2.5 + 2.5) % 5.0) - 2.5) < 0.1] -= 0.01  # transiti P=5, t0=2.5, 0.2 d
    return t, flux


def test_views_for_koi_returns_views_with_expected_shapes():
    t, flux = _curve()
    row = prepare_koi_table(_multi_table()).iloc[0]
    v = views_for_koi(t, flux, row, siblings=[])
    assert v.global_view.shape == (GLOBAL_BINS,) and v.local_view.shape == (LOCAL_BINS,)
    assert abs(int(np.argmin(v.local_view)) - LOCAL_BINS // 2) < 0.15 * LOCAL_BINS


def test_views_for_koi_skips_missing_snr_and_too_few_transits():
    t, flux = _curve()
    row = prepare_koi_table(_multi_table()).iloc[0].copy()
    short_t, short_f = t[t < 12.0], flux[t < 12.0]  # solo le epoche 2.5 e 7.5
    with pytest.raises(SkipKOI, match="transiti_osservati"):
        views_for_koi(short_t, short_f, row, siblings=[])
    row["snr"] = np.nan
    with pytest.raises(SkipKOI, match="snr_mancante"):
        views_for_koi(t, flux, row, siblings=[])
