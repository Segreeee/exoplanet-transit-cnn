"""Costruisce il dataset di viste (npz) da curve di luce sintetiche o da Kepler.

    python scripts/build_dataset.py --source synthetic --n 400 --workers 2 \
        --out data/processed/synthetic.npz
    python scripts/build_dataset.py --source kepler --koi data/raw/koi.csv \
        --cache data/raw/kepler --out data/processed/kepler.npz --workers 4

Per `kepler` servono prima i file prodotti da scripts/download_kepler.py.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from exotransit.koi import prepare_koi_table, row_to_ephemeris
from exotransit.preprocess import process_light_curve
from exotransit.synthetic import make_dataset


def _views_to_arrays(views):
    return views.global_view, views.local_view, views.extra


def _process_synthetic(args):
    ex, max_periods = args
    try:
        views, _ = process_light_curve(ex.time, ex.flux, max_periods=max_periods)
    except Exception as exc:  # una curva difettosa non deve fermare tutto il batch
        return None, f"star {ex.star_id}: {type(exc).__name__}: {exc}"
    return (*_views_to_arrays(views), ex.label, ex.star_id, ex.kind), None


def _process_kepler(args):
    row, cache_dir = args
    path = Path(cache_dir) / f"{int(row['kepid'])}.npz"
    if not path.exists():
        return None, f"{row['name']}: curva non scaricata ({path.name})"
    try:
        with np.load(path) as d:
            time, flux = d["time"], d["flux"]
        views, _ = process_light_curve(time, flux, ephemeris=row_to_ephemeris(row))
    except Exception as exc:
        return None, f"{row['name']}: {type(exc).__name__}: {exc}"
    return (*_views_to_arrays(views), int(row["label"]), int(row["kepid"]), row["name"]), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "kepler"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n", type=int, default=400, help="(synthetic) numero di esempi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-periods", type=int, default=5000, help="(synthetic) griglia BLS")
    ap.add_argument("--koi", help="(kepler) CSV della tabella KOI")
    ap.add_argument("--cache", help="(kepler) cartella con i {kepid}.npz")
    ap.add_argument("--include-candidates", action="store_true")
    ap.add_argument("--limit", type=int, help="(kepler) elabora solo i primi N KOI")
    args = ap.parse_args()

    if args.source == "synthetic":
        jobs = [(ex, args.max_periods) for ex in make_dataset(args.n, seed=args.seed)]
        worker = _process_synthetic
    else:
        if not (args.koi and args.cache):
            ap.error("--koi e --cache sono obbligatori con --source kepler")
        table = prepare_koi_table(pd.read_csv(args.koi), args.include_candidates)
        if args.limit:
            table = table.head(args.limit)
        jobs = [(row, args.cache) for _, row in table.iterrows()]
        worker = _process_kepler

    ok, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (res, err) in enumerate(pool.map(worker, jobs, chunksize=4), 1):
            (ok if res is not None else errors).append(res if res is not None else err)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} elaborati ({len(errors)} scartati)", flush=True)

    if not ok:
        print("Nessun esempio elaborato con successo.", file=sys.stderr)
        for e in errors[:10]:
            print("  -", e, file=sys.stderr)
        return 1

    g, l_, extra, y, groups, names = zip(*ok, strict=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        global_view=np.stack(g),
        local_view=np.stack(l_),
        extra=np.stack(extra),
        y=np.array(y, dtype=np.int64),
        groups=np.array(groups, dtype=np.int64),
        names=np.array(names),
    )
    print(f"Salvato {out} con {len(y)} esempi ({int(sum(y))} positivi), {len(errors)} scartati.")
    for e in errors[:5]:
        print("  scartato:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
