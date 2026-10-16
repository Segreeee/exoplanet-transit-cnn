"""Tabella TOI (TESS Objects of Interest, NASA Exoplanet Archive) -> etichette e effemeridi.

Stessa interfaccia di `koi.prepare_koi_table`, cosi' `koi.views_for_koi` e `build_dataset`
funzionano anche per TESS. TESS in due righe: satellite NASA (dal 2018) che osserva quasi tutto
il cielo a settori di ~27 giorni; ogni stella ha pochi transiti rispetto ai 4 anni di Kepler.
Differenze da tenere a mente:
- il tempo e' in BTJD (= BJD - 2457000), la tabella TOI da' BJD;
- la tabella TOI NON ha una colonna SNR: `snr` vale 0.0 (segnaposto; `no_scalar` la ignora);
- le disposizioni sono TFOP (comunita') e non del team Kepler: CP/KP = pianeta confermato/noto
  (1), FP = falso positivo, FA = falso allarme (0); PC/APC (candidati) sono scartati come i
  CANDIDATE Kepler.

Colonna `kepid` = ID della stella: per TESS e' il TIC (`tid`).
"""

from __future__ import annotations

import pandas as pd

TOI_BTJD_OFFSET = 2457000.0
TOI_COLUMNS = (
    "tid",
    "toi",
    "tfopwg_disp",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandurh",
    "pl_trandep",
)
TOI_TAP_QUERY = "select " + ",".join(TOI_COLUMNS) + " from toi"

_LABELS = {"CP": 1, "KP": 1, "FP": 0, "FA": 0}
_CANDIDATES = {"PC": 1, "APC": 1}


def prepare_toi_table(df: pd.DataFrame, include_candidates: bool = False) -> pd.DataFrame:
    """Filtra e converte la tabella TOI: colonne kepid(=TIC), name, label, period, t0 (BTJD),
    duration (giorni), depth (frazione), snr (0.0). Con `include_candidates` PC/APC diventano
    positivi (serve a mascherare i loro transiti, non per l'addestramento)."""
    missing = [c for c in TOI_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nella tabella TOI: {missing}")
    out = pd.DataFrame(
        {
            "kepid": df["tid"].astype("int64"),
            "name": "TOI-" + df["toi"].astype(str),
            "label": df["tfopwg_disp"].map(
                {**_LABELS, **(_CANDIDATES if include_candidates else {})}
            ),
            "period": df["pl_orbper"].astype(float),
            "t0": df["pl_tranmid"].astype(float) - TOI_BTJD_OFFSET,
            "duration": df["pl_trandurh"].astype(float) / 24.0,  # ore -> giorni
            "depth": df["pl_trandep"].astype(float) / 1e6,  # ppm -> frazione
            "snr": 0.0,
        }
    )
    out = out.dropna(subset=["label", "period", "t0", "duration", "depth"])
    out = out[(out["period"] > 0) & (out["duration"] > 0) & (out["depth"] > 0)]
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)
