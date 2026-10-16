"""Scarica la tabella TOI (NASA Exoplanet Archive) e le curve di luce TESS SPOC 2-min (MAST).

    pip install -e ".[kepler]"                       # lightkurve serve anche per TESS
    python scripts/download_tess.py --limit 1000 --workers 8

Per ogni stella: PDCSAP di tutti i settori (`stitch` normalizza ogni settore), poi binning a
10 minuti per risparmiare disco e tempo (i transiti durano ore; la cadenza originale e' 2 minuti)
e salvataggio di time (BTJD) e flux in data/raw/tess/{tic}.npz. Cache FITS cancellata per stella.
Le stelle senza SPOC 2-min (solo FFI) falliscono e finiscono in `--failed`.

Attenzione all'epoca: `pl_tranmid` e' un'epoca di riferimento del catalogo; per dati lontani da essa
l'errore sul periodo si accumula (ephemeris drift). E' un limite del catalogo, non lo correggiamo.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from exotransit.lcdownload import (
    MIN_FREE_BYTES,
    clear_star_cache,
    download_with_cache_recovery,
    has_free_space,
    purge_empty_cache_files,
    run_downloads,
    save_npz_atomic,
)
from exotransit.toi import TOI_TAP_QUERY, prepare_toi_table

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
BIN_MINUTES = 10


def fetch_toi_table(dest: Path) -> pd.DataFrame:
    if dest.exists():
        return pd.read_csv(dest)
    r = requests.get(TAP_URL, params={"query": TOI_TAP_QUERY, "format": "csv"}, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(r.text, encoding="utf-8")
    return pd.read_csv(io.StringIO(r.text))


def download_light_curve(tic: int, cache_dir: Path) -> bool:
    import astropy.units as u
    import lightkurve as lk  # import locale: dipendenza opzionale

    path = cache_dir / f"{tic}.npz"
    if path.exists():
        return True
    found = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120)
    if len(found) == 0:
        return False
    lk_cache = Path(lk.config.get_cache_dir())
    try:
        purge = lambda: purge_empty_cache_files(lk_cache, tic, mission="TESS")  # noqa: E731
        collection = download_with_cache_recovery(found, purge)
        lc = collection.stitch().remove_nans()  # stitch normalizza ogni settore
        lc = lc.bin(time_bin_size=BIN_MINUTES * u.min).remove_nans()
        save_npz_atomic(
            path,
            time=np.asarray(lc.time.value, dtype=float),
            flux=np.asarray(lc.flux.value, dtype=float),
        )
    finally:
        clear_star_cache(lk_cache, tic, mission="TESS")
    return True


def fetch_star(job) -> tuple[int, str | None]:
    """(tic, errore o None). Gira in un processo a parte (vedi `lcdownload.run_downloads`)."""
    tic, cache_dir, min_free = job
    cache = Path(cache_dir)
    if not has_free_space(cache, min_free):
        return tic, "disco quasi pieno: download fermato"
    try:
        ok = download_light_curve(tic, cache)
    except Exception as exc:  # rete instabile: continua e riprova alla prossima esecuzione
        return tic, f"{type(exc).__name__}: {exc}"
    return tic, None if ok else "nessuna curva SPOC 2-min"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--toi", default="data/raw/toi.csv")
    ap.add_argument("--cache", default="data/raw/tess")
    ap.add_argument("--limit", type=int, help="scarica solo N stelle (dopo il mescolamento)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="processi paralleli (4 se MAST limita)")
    ap.add_argument("--failed", default="data/raw/tess_failed.txt")
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_BYTES / 1e9,
        help="si ferma sotto questo spazio",
    )
    args = ap.parse_args()

    table = prepare_toi_table(fetch_toi_table(Path(args.toi)))
    print(table["label"].value_counts().rename({1: "positivi", 0: "falsi positivi"}).to_string())
    tics = table["kepid"].drop_duplicates().sample(frac=1.0, random_state=args.seed).tolist()
    if args.limit:
        tics = tics[: args.limit]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    jobs = [(int(t), str(cache), args.min_free_gb * 1e9) for t in tics]
    n_ok, n_failed = run_downloads(fetch_star, jobs, args.workers, args.failed)
    print(f"Fatto. {n_ok} scaricate, {n_failed} fallite (elenco: {args.failed}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
