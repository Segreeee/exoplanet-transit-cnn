"""Tabella KOI (Kepler Objects of Interest) della NASA Exoplanet Archive -> etichette e effemeridi.

Il download di rete vive in scripts/download_kepler.py; qui c'e' solo la logica pura e
testabile. Le colonne attese sono quelle della tabella `cumulative` della Exoplanet Archive:
kepid, kepoi_name, koi_disposition, koi_period (giorni), koi_time0bk (BKJD),
koi_duration (ORE), koi_depth (ppm), koi_model_snr.
"""

from __future__ import annotations

import pandas as pd

from exotransit.preprocess import Ephemeris

REQUIRED_COLUMNS = (
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_model_snr",
)

TAP_QUERY = "select " + ",".join(REQUIRED_COLUMNS) + " from cumulative"


def prepare_koi_table(df: pd.DataFrame, include_candidates: bool = False) -> pd.DataFrame:
    """Filtra e converte la tabella KOI in una tabella pronta per il training.

    Default: CONFIRMED = 1, FALSE POSITIVE = 0, CANDIDATE scartati (etichetta incerta).
    Con `include_candidates=True` i CANDIDATE diventano positivi: piu' dati, piu' rumore
    nelle etichette. Righe senza effemeride valida vengono eliminate.
    Colonne in uscita: kepid, name, label, period, t0, duration (giorni), depth (frazione), snr.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nella tabella KOI: {missing}")

    labels = {"CONFIRMED": 1, "FALSE POSITIVE": 0}
    if include_candidates:
        labels["CANDIDATE"] = 1
    out = pd.DataFrame(
        {
            "kepid": df["kepid"].astype("int64"),
            "name": df["kepoi_name"],
            "label": df["koi_disposition"].map(labels),
            "period": df["koi_period"].astype(float),
            "t0": df["koi_time0bk"].astype(float),
            "duration": df["koi_duration"].astype(float) / 24.0,  # ore -> giorni
            "depth": df["koi_depth"].astype(float) / 1e6,  # ppm -> frazione
            "snr": df["koi_model_snr"].astype(float),
        }
    )
    out = out.dropna(subset=["label", "period", "t0", "duration", "depth"])
    out = out[(out["period"] > 0) & (out["duration"] > 0) & (out["depth"] > 0)]
    out["label"] = out["label"].astype(int)
    out["snr"] = out["snr"].fillna(0.0)
    return out.reset_index(drop=True)


def row_to_ephemeris(row) -> Ephemeris:
    return Ephemeris(
        period=float(row["period"]),
        t0=float(row["t0"]),
        duration=float(row["duration"]),
        depth=float(row["depth"]),
        snr=float(row["snr"]),
    )
