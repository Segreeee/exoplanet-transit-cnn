"""Scarica la tabella KOI (NASA Exoplanet Archive) e le curve di luce Kepler (MAST).

    pip install -e ".[kepler]"
    python scripts/download_kepler.py --limit 50          # PARTI DA POCHI: ogni stella pesa MB
    python scripts/download_kepler.py                     # tutto: ore e molti GB

STATO: verificato sul set completo (6424 stelle). Gli helper di cache e disco stanno in
`exotransit.lcdownload` (condivisi con TESS).

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
from exotransit.lcdownload import (
    MIN_FREE_BYTES,
    clear_star_cache,
    download_with_cache_recovery,
    has_free_space,
    purge_empty_cache_files,
    run_downloads,
    save_npz_atomic,
)

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
    lk_cache = Path(lk.config.get_cache_dir())
    try:
        purge = lambda: purge_empty_cache_files(lk_cache, kepid)  # noqa: E731
        collection = download_with_cache_recovery(found, purge)
        lc = collection.stitch().remove_nans()  # stitch normalizza ogni trimestre
        save_npz_atomic(
            path,
            time=np.asarray(lc.time.value, dtype=float),
            flux=np.asarray(lc.flux.value, dtype=float),
        )
    finally:
        clear_star_cache(lk_cache, kepid)  # i FITS grezzi non servono piu': sono ~14x il .npz
    return True


def fetch_star(job) -> tuple[int, str | None]:
    """(kepid, errore o None). Gira in un processo a parte: `lightkurve.utils.suppress_stdout`
    sostituisce `sys.stdout` globale, e con piu' THREAD lo lascia chiuso (verificato: crash)."""
    kepid, cache_dir = job
    cache = Path(cache_dir)
    if not has_free_space(cache, MIN_FREE_BYTES):
        return kepid, "disco quasi pieno: download fermato"
    try:
        ok = download_light_curve(kepid, cache)
    except Exception as exc:  # rete instabile: continua e riprova alla prossima esecuzione
        return kepid, f"{type(exc).__name__}: {exc}"
    return kepid, None if ok else "nessuna curva Kepler a cadenza lunga"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--koi", default="data/raw/koi.csv")
    ap.add_argument("--cache", default="data/raw/kepler")
    ap.add_argument("--limit", type=int, help="scarica solo le prime N stelle")
    ap.add_argument("--include-candidates", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="mescola le stelle prima del limite")
    ap.add_argument("--workers", type=int, default=4, help="processi paralleli (2 se MAST limita)")
    ap.add_argument("--failed", default="data/raw/kepler_failed.txt", help="elenco dei fallimenti")
    args = ap.parse_args()

    table = prepare_koi_table(fetch_koi_table(Path(args.koi)), args.include_candidates)
    print(table["label"].value_counts().rename({1: "positivi", 0: "falsi positivi"}).to_string())

    kepids = table["kepid"].drop_duplicates().sample(frac=1.0, random_state=args.seed).tolist()
    if args.limit:
        kepids = kepids[: args.limit]

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(int(k), str(cache)) for k in kepids]
    n_ok, n_failed = run_downloads(fetch_star, jobs, args.workers, args.failed)
    print(f"Fatto. {n_ok} scaricate, {n_failed} fallite (elenco: {args.failed}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
