"""Worker e report di scripts/build_dataset.py --source kepler (fixture minuscole, nessuna rete)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from exotransit.koi import prepare_koi_table

_spec = importlib.util.spec_from_file_location(
    "build_dataset", Path(__file__).parents[1] / "scripts" / "build_dataset.py"
)
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)


def _row(snr=50.0):
    df = pd.DataFrame(
        {
            "kepid": [7],
            "kepoi_name": ["K7.01"],
            "koi_disposition": ["CONFIRMED"],
            "koi_period": [5.0],
            "koi_time0bk": [2.5],
            "koi_duration": [4.8],  # ore
            "koi_depth": [10000.0],  # ppm
            "koi_model_snr": [snr],
        }
    )
    return prepare_koi_table(df).iloc[0]


def _save_curve(cache: Path, t_end: float):
    rng = np.random.default_rng(0)
    t = np.arange(0.0, t_end, 0.02)
    flux = 1.0 + rng.normal(0.0, 1e-4, t.size)
    flux[np.abs((t % 5.0) - 2.5) < 0.1] -= 0.01
    np.savez_compressed(cache / "7.npz", time=t, flux=flux)


def test_worker_reports_reason_for_each_kind_of_discard(tmp_path):
    job = (_row(), [], str(tmp_path), 3)
    res, err = bd._process_kepler(job)
    assert res is None and err.startswith("curva_non_scaricata:")

    _save_curve(tmp_path, t_end=12.0)  # solo 2 transiti osservati
    res, err = bd._process_kepler(job)
    assert res is None and err.startswith("transiti_osservati<3:")

    _save_curve(tmp_path, t_end=100.0)
    res, err = bd._process_kepler((_row(snr=np.nan), [], str(tmp_path), 3))
    assert res is None and err.startswith("snr_mancante:")


def test_worker_returns_views_label_group_and_name(tmp_path):
    _save_curve(tmp_path, t_end=100.0)
    res, err = bd._process_kepler((_row(), [], str(tmp_path), 3))
    assert err is None
    g, loc, extra, label, kepid, name = res
    assert g.shape == (2001,) and loc.shape == (201,) and extra.shape == (4,)
    assert (label, kepid, name) == (1, 7, "K7.01")


def test_dataset_report_counts_classes_stars_and_reasons():
    y = np.array([1, 0, 0, 1])
    groups = np.array([10, 10, 11, 12])  # la stella 10 ha 2 KOI
    errors = ["snr_mancante: K1.01", "snr_mancante: K2.01", "transiti_osservati<3: K3.01"]
    r = bd.dataset_report(y, groups, errors)
    assert r["n_examples"] == 4 and r["n_stars"] == 3
    assert r["examples_per_class"] == {"positivi": 2, "falsi_positivi": 2}
    assert r["stars_per_class"] == {"positivi": 2, "falsi_positivi": 2}
    assert r["n_stars_with_multiple_kois"] == 1
    assert r["discarded"] == {"n": 3, "by_reason": {"snr_mancante": 2, "transiti_osservati<3": 1}}
