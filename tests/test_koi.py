import numpy as np
import pandas as pd
import pytest

from exotransit.koi import prepare_koi_table, row_to_ephemeris


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
    assert t.loc[1, "snr"] == 0.0  # NaN -> 0


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
