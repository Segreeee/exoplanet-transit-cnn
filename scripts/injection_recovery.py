"""Fase 4.3: injection-recovery. Inietta transiti sintetici in curve Kepler reali e "tranquille".

    python scripts/injection_recovery.py --data data/processed/kepler.npz \
        --cnn-report reports/kepler_cnn.json --models-dir models/kepler

Per ogni seed si usano solo le stelle del TEST di quel seed (il modello non le ha viste), si tiene
la meta' con lo scatter punto-a-punto piu' basso, si eliminano i transiti di TUTTI i loro KOI e si
inietta un pianeta (profilo trapezoidale, durata ~ P^(1/3)) su una griglia profondita' x periodo.
Completezza = quota di iniezioni con punteggio >= soglia scelta sulla validazione.

Limiti: l'effemeride e' nota, quindi si misura il CLASSIFICATORE, non la rivelazione (niente BLS);
le iniezioni sono pianeti per costruzione: nessuna stima del tasso di falsi positivi.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from exotransit.experiments import load_model
from exotransit.koi import prepare_koi_table, row_to_ephemeris
from exotransit.perturb import inject_transit, point_scatter, transit_duration_days
from exotransit.preprocess import (
    Ephemeris,
    count_observed_transits,
    process_light_curve,
    transit_mask,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEPTHS_PPM = (200, 500, 1000, 2000, 5000, 10000)
PERIODS = (2, 4, 8, 16, 32, 64, 128)


def _inject_job(job):
    """-> (g, l, extra) oppure None se i transiti osservati sono < 3."""
    path, siblings, period, depth, rng_seed = job
    with np.load(path) as d:
        time, flux = d["time"], d["flux"]
    rng = np.random.default_rng(rng_seed)
    t0 = float(time[0] + rng.uniform(0.0, period))
    duration = transit_duration_days(period)
    keep = np.ones(time.size, dtype=bool)
    for other in siblings:  # via i transiti dei KOI veri: la curva resta "tranquilla"
        keep &= ~transit_mask(time, other.period, other.t0, other.duration)
    time, flux = time[keep], flux[keep]
    flux = inject_transit(time, flux, period, t0, depth, duration)
    if count_observed_transits(time, period, t0, duration) < 3:
        return None
    eph = Ephemeris(period=period, t0=t0, duration=duration, depth=depth, snr=0.0)
    views, _ = process_light_curve(time, flux, ephemeris=eph)
    return views.global_view, views.local_view, views.extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cnn-report", default="reports/kepler_cnn.json")
    ap.add_argument("--models-dir", default="models/kepler")
    ap.add_argument("--variant", default="no_scalar")
    ap.add_argument("--koi", default="data/raw/koi.csv")
    ap.add_argument("--cache", default="data/raw/kepler")
    ap.add_argument("--report", default="reports/kepler_injection.json")
    ap.add_argument("--figure", default="reports/figures/injection_recovery.png")
    ap.add_argument("--per-cell", type=int, default=12, help="stelle per cella e per seed")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    with np.load(args.data) as d:
        groups = d["groups"]
    runs = [r for r in json.load(open(args.cnn_report))["runs"] if r["variant"] == args.variant]
    all_kois = prepare_koi_table(pd.read_csv(args.koi), include_candidates=True)
    ephs = {k: [row_to_ephemeris(r) for _, r in g.iterrows()] for k, g in all_kois.groupby("kepid")}

    # 1) stelle tranquille per seed e lavori
    jobs, keys = [], []
    for r in runs:
        stars = sorted(set(groups[np.asarray(r["test_idx"])].tolist()))
        scatter = {}
        for k in stars:
            with np.load(Path(args.cache) / f"{k}.npz") as c:
                scatter[k] = point_scatter(c["flux"])
        quiet = sorted(stars, key=scatter.get)[: len(stars) // 2]
        rng = np.random.default_rng(1000 + r["seed"])
        for depth in DEPTHS_PPM:
            for period in PERIODS:
                for star in rng.choice(quiet, size=args.per_cell, replace=False):
                    seed_j = int(rng.integers(1 << 31))
                    path = str(Path(args.cache) / f"{int(star)}.npz")
                    jobs.append(
                        (path, ephs.get(int(star), []), float(period), depth * 1e-6, seed_j)
                    )
                    keys.append((r["seed"], depth, period))
    print(f"{len(jobs)} iniezioni su {len(runs)} seed", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_inject_job, jobs, chunksize=8))

    # 2) punteggi con il modello di ogni seed
    rows = []
    for r in runs:
        model, thr = load_model(Path(args.models_dir) / f"{args.variant}_seed{r['seed']}.pt")
        sel = [i for i, k in enumerate(keys) if k[0] == r["seed"]]
        ok = [i for i in sel if results[i] is not None]
        scores = {}
        if ok:
            g = np.stack([results[i][0] for i in ok])
            l_ = np.stack([results[i][1] for i in ok])
            e = np.stack([results[i][2] for i in ok])
            scores = dict(zip(ok, model.predict_proba(g, l_, e), strict=True))
        for i in sel:
            rows.append(
                {
                    "seed": r["seed"],
                    "depth_ppm": keys[i][1],
                    "period": keys[i][2],
                    "valid": i in scores,
                    "recovered": bool(scores[i] >= thr) if i in scores else False,
                }
            )
    df = pd.DataFrame(rows)

    # 3) completezza per cella: media +- std sui seed (solo iniezioni valide)
    per_seed = (
        df[df.valid]
        .groupby(["seed", "depth_ppm", "period"])
        .agg(completeness=("recovered", "mean"), n=("valid", "size"))
        .reset_index()
    )
    cell = (
        per_seed.groupby(["depth_ppm", "period"])
        .agg(mean=("completeness", "mean"), std=("completeness", "std"), n=("n", "sum"))
        .reset_index()
    )
    n_inj = df.groupby(["depth_ppm", "period"]).size().rename("n_injected").reset_index()
    cell = n_inj.merge(cell, how="left", on=["depth_ppm", "period"])
    grid = cell.pivot(index="depth_ppm", columns="period", values="mean")
    print("\nCompletezza (media sui seed; NaN = meno di 3 transiti osservati):")
    print(grid.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(grid.values, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
    for (i, j), v in np.ndenumerate(grid.values):
        ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", color="w")
    ax.set(xlabel="periodo (giorni)", ylabel="profondita' (ppm)")
    ax.set_title("Completezza del classificatore (iniezioni in curve Kepler tranquille)")
    fig.colorbar(im, label="quota >= soglia")
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=110)
    plt.close(fig)

    report = {
        "args": vars(args),
        "depths_ppm": list(DEPTHS_PPM),
        "periods": list(PERIODS),
        "n_injections": len(df),
        "n_valid": int(df.valid.sum()),
        "cells": cell.to_dict("records"),
        "per_seed": per_seed.to_dict("records"),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Salvato {args.report} e {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
