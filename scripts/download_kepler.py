"""Scarica la tabella KOI (NASA Exoplanet Archive) e le curve di luce Kepler (MAST).

    pip install -e ".[kepler]"
    python scripts/download_kepler.py --limit 50          # PARTI DA POCHI: ogni stella pesa MB
    python scripts/download_kepler.py                     # tutto: ore e molti GB

STATO: scritto ma NON verificato contro i server reali (l'ambiente in cui e' stato generato
non poteva raggiungere NASA/MAST). Se fallisce, e' il primo compito della Fase 1: usa la skill
`diagnose` e correggi lo script, aggiungendo un test per la parte di logica pura.

Nota: scarica PDCSAP (flusso corretto per le sistematiche) con `lightkurve`, normalizza ogni
trimestre e salva solo time/flux in data/raw/kepler/{kepid}.npz.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from exotransit.koi import TAP_QUERY, prepare_koi_table

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def fetch_koi_table(dest: Path) -> pd.DataFrame:
    if dest.exists():
        return pd.read_csv(dest)
    r = requests.get(TAP_URL, params={"query": TAP_QUERY, "format": "csv"}, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(r.text)
    return pd.read_csv(io.StringIO(r.text))


def download_light_curve(kepid: int, cache_dir: Path) -> bool:
    import lightkurve as lk  # import locale: dipendenza opzionale

    path = cache_dir / f"{kepid}.npz"
    if path.exists():
        return True
    found = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler", cadence="long")
    if len(found) == 0:
        return False
    lc = found.download_all().stitch().remove_nans()  # stitch normalizza ogni trimestre
    np.savez_compressed(
        path,
        time=np.asarray(lc.time.value, dtype=float),
        flux=np.asarray(lc.flux.value, dtype=float),
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--koi", default="data/raw/koi.csv")
    ap.add_argument("--cache", default="data/raw/kepler")
    ap.add_argument("--limit", type=int, help="scarica solo le prime N stelle")
    ap.add_argument("--include-candidates", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="mescola le stelle prima del limite")
    args = ap.parse_args()

    table = prepare_koi_table(fetch_koi_table(Path(args.koi)), args.include_candidates)
    print(table["label"].value_counts().rename({1: "positivi", 0: "falsi positivi"}).to_string())

    kepids = table["kepid"].drop_duplicates().sample(frac=1.0, random_state=args.seed).tolist()
    if args.limit:
        kepids = kepids[: args.limit]

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, kepid in enumerate(kepids, 1):
        try:
            ok = download_light_curve(int(kepid), cache)
        except Exception as exc:  # rete instabile: continua e riprova alla prossima esecuzione
            ok, _ = False, print(f"  KIC {kepid}: {type(exc).__name__}: {exc}")
        if not ok:
            failed.append(kepid)
        print(f"[{i}/{len(kepids)}] KIC {kepid} {'ok' if ok else 'FALLITA'}", flush=True)
    print(f"Fatto. {len(kepids) - len(failed)} scaricate, {len(failed)} fallite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
