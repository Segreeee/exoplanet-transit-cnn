"""Tabella KOI (Kepler Objects of Interest) della NASA Exoplanet Archive -> etichette e effemeridi.

Il download di rete vive in scripts/download_kepler.py; qui c'e' solo la logica pura e
testabile. Le colonne attese sono quelle della tabella `cumulative` della Exoplanet Archive:
kepid, kepoi_name, koi_disposition, koi_period (giorni), koi_time0bk (BKJD),
koi_duration (ORE), koi_depth (ppm), koi_model_snr.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from exotransit.preprocess import (
    CandidateViews,
    Ephemeris,
    count_observed_transits,
    process_light_curve,
)

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
    return out.reset_index(drop=True)


def row_to_ephemeris(row) -> Ephemeris:
    return Ephemeris(
        period=float(row["period"]),
        t0=float(row["t0"]),
        duration=float(row["duration"]),
        depth=float(row["depth"]),
        snr=float(row["snr"]),
    )


class SkipKOI(Exception):
    """Candidato da scartare; `reason` e' una chiave stabile per i conteggi nel report."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def sibling_ephemerides(all_kois: pd.DataFrame, kepid: int, name: str) -> list[Ephemeris]:
    """Effemeridi degli altri KOI della stessa stella (di qualunque disposizione).

    `all_kois` va costruita con `prepare_koi_table(..., include_candidates=True)`: i CANDIDATE
    non entrano nel training ma i loro transiti sono comunque nella curva e vanno mascherati.
    """
    rows = all_kois[(all_kois["kepid"] == kepid) & (all_kois["name"] != name)]
    return [row_to_ephemeris(r) for _, r in rows.iterrows()]


def views_for_koi(
    time: np.ndarray,
    flux: np.ndarray,
    row,
    siblings: Sequence[Ephemeris],
    min_transits: int = 3,
) -> CandidateViews:
    """Viste di un KOI da una curva grezza, o `SkipKOI` con il motivo dello scarto.

    Nessun taglio sull'SNR (si filtra dopo, da `extra[:, 3]`), ma un SNR mancante e' uno scarto
    esplicito: non va mai trattato come 0.
    """
    if not np.isfinite(row["snr"]):
        raise SkipKOI("snr_mancante")
    eph = row_to_ephemeris(row)
    t, f = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    if count_observed_transits(t[ok], eph.period, eph.t0, eph.duration) < min_transits:
        raise SkipKOI(f"transiti_osservati<{min_transits}")
    views, _ = process_light_curve(t, f, ephemeris=eph, exclude=siblings)
    return views
