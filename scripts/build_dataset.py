"""Costruisce il dataset di viste (npz) da curve di luce sintetiche o da Kepler.

    python scripts/build_dataset.py --source synthetic --n 400 --workers 2 \
        --out data/processed/synthetic.npz
    python scripts/build_dataset.py --source kepler --koi data/raw/koi.csv \
        --cache data/raw/kepler --out data/processed/kepler.npz --workers 4

Per `kepler` servono prima i file prodotti da scripts/download_kepler.py. Un esempio per KOI
(gruppo = kepid); i transiti degli altri KOI della stessa stella vengono mascherati. Scarti:
curva mancante, SNR mancante, meno di --min-transits transiti osservati. Nessun taglio sull'SNR.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from exotransit.koi import SkipKOI, prepare_koi_table, sibling_ephemerides, views_for_koi
from exotransit.preprocess import process_light_curve
from exotransit.synthetic import make_dataset


def _views_to_arrays(views):
    return views.global_view, views.local_view, views.extra


def _process_synthetic(args):
    ex, max_periods = args
    try:
        views, _ = process_light_curve(ex.time, ex.flux, max_periods=max_periods)
    except Exception as exc:  # una curva difettosa non deve fermare tutto il batch
        return None, f"errore_{type(exc).__name__}: star {ex.star_id}: {exc}"
    return (*_views_to_arrays(views), ex.label, ex.star_id, ex.kind), None


def _process_kepler(args):
    row, siblings, cache_dir, min_transits = args
    path = Path(cache_dir) / f"{int(row['kepid'])}.npz"
    if not path.exists():
        return None, f"curva_non_scaricata: {row['name']}"
    try:
        with np.load(path) as d:
            time, flux = d["time"], d["flux"]
        views = views_for_koi(time, flux, row, siblings, min_transits)
    except SkipKOI as skip:
        return None, f"{skip.reason}: {row['name']}"
    except Exception as exc:
        return None, f"errore_{type(exc).__name__}: {row['name']}: {exc}"
    return (*_views_to_arrays(views), int(row["label"]), int(row["kepid"]), row["name"]), None


def dataset_report(y, groups, errors) -> dict:
    """Conteggi del dataset costruito e degli scarti (per motivo: il testo prima dei ':')."""
    y, groups = np.asarray(y), np.asarray(groups)
    per_star = Counter(groups.tolist())
    names = {1: "positivi", 0: "falsi_positivi"}
    return {
        "n_examples": int(y.size),
        "n_stars": len(per_star),
        "examples_per_class": {names[c]: int((y == c).sum()) for c in (1, 0)},
        "stars_per_class": {names[c]: int(np.unique(groups[y == c]).size) for c in (1, 0)},
        "n_stars_with_multiple_kois": sum(n > 1 for n in per_star.values()),
        "discarded": {
            "n": len(errors),
            "by_reason": dict(Counter(e.split(":")[0] for e in errors)),
        },
    }


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
    ap.add_argument(
        "--min-transits", type=int, default=3, help="(kepler) transiti osservati minimi"
    )
    ap.add_argument("--report", help="JSON con config, versioni e conteggi del dataset")
    args = ap.parse_args()

    if args.source == "synthetic":
        jobs = [(ex, args.max_periods) for ex in make_dataset(args.n, seed=args.seed)]
        worker = _process_synthetic
    else:
        if not (args.koi and args.cache):
            ap.error("--koi e --cache sono obbligatori con --source kepler")
        raw = pd.read_csv(args.koi)
        all_kois = prepare_koi_table(raw, include_candidates=True)  # per mascherare i fratelli
        table = prepare_koi_table(raw, args.include_candidates)
        if args.limit:
            table = table.head(args.limit)
        jobs = [
            (
                row,
                sibling_ephemerides(all_kois, int(row["kepid"]), row["name"]),
                args.cache,
                args.min_transits,
            )
            for _, row in table.iterrows()
        ]
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
    if args.report:
        report = {
            "args": vars(args),
            "versions": {p: version(p) for p in ("numpy", "pandas", "astropy", "scipy")},
            **dataset_report(y, groups, errors),
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
