"""Fase 4.4: robustezza del classificatore a meno dati e a piu' rumore.

    python scripts/robustness.py --data data/processed/kepler.npz \
        --cnn-report reports/kepler_cnn.json --models-dir models/kepler

Sui KOI di TEST di ogni seed si perturbano le curve grezze (baseline troncata ai primi 365 / 730
giorni, rumore che porta lo scatter punto-a-punto a x2 / x4), si rifanno le viste e si
ri-valuta il modello di quel seed alla SUA soglia di validazione. I KOI con meno di 3 transiti
osservati dopo la perturbazione vengono scartati (e contati): il confronto con la curva non
perturbata e' fatto sullo stesso sottoinsieme.
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
from exotransit.koi import SkipKOI, prepare_koi_table, sibling_ephemerides, views_for_koi
from exotransit.metrics import evaluate_scores
from exotransit.perturb import perturb_light_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PERTURBATIONS = (
    ("none", 0.0),
    ("truncate", 365.0),
    ("truncate", 730.0),
    ("noise", 2.0),
    ("noise", 4.0),
)


def _label(kind: str, level: float) -> str:
    return {"none": "originale", "truncate": f"primi {level:.0f} d", "noise": f"rumore x{level:g}"}[
        kind
    ]


def _views_job(job):
    """-> (g, l, extra) oppure None (scartato)."""
    path, row, siblings, kind, level, rng_seed = job
    with np.load(path) as d:
        time, flux = d["time"], d["flux"]
    if kind != "none":
        time, flux = perturb_light_curve(time, flux, kind, level, np.random.default_rng(rng_seed))
    try:
        v = views_for_koi(time, flux, row, siblings, min_transits=3)
    except SkipKOI:
        return None
    return v.global_view, v.local_view, v.extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cnn-report", default="reports/kepler_cnn.json")
    ap.add_argument("--models-dir", default="models/kepler")
    ap.add_argument("--variant", default="no_scalar")
    ap.add_argument("--koi", default="data/raw/koi.csv")
    ap.add_argument("--cache", default="data/raw/kepler")
    ap.add_argument("--report", default="reports/kepler_robustness.json")
    ap.add_argument("--figure", default="reports/figures/robustness.png")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    with np.load(args.data) as d:
        names, y = d["names"], d["y"]
    runs = [r for r in json.load(open(args.cnn_report))["runs"] if r["variant"] == args.variant]
    raw = pd.read_csv(args.koi)
    table = prepare_koi_table(raw).set_index("name", drop=False)
    all_kois = prepare_koi_table(raw, include_candidates=True)

    # lavori unici per (KOI, perturbazione): lo stesso KOI puo' stare nel test di piu' seed
    wanted = sorted({names[i] for r in runs for i in r["test_idx"]})
    rng = np.random.default_rng(0)
    jobs, job_keys = [], []
    for name in wanted:
        row = table.loc[name]
        sib = sibling_ephemerides(all_kois, int(row["kepid"]), name)
        path = str(Path(args.cache) / f"{int(row['kepid'])}.npz")
        for kind, level in PERTURBATIONS:
            jobs.append((path, row, sib, kind, level, int(rng.integers(1 << 31))))
            job_keys.append((name, kind, level))
    print(f"{len(jobs)} elaborazioni ({len(wanted)} KOI x {len(PERTURBATIONS)})", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        out = dict(zip(job_keys, pool.map(_views_job, jobs, chunksize=8), strict=True))

    rows = []
    for r in runs:
        model, thr = load_model(Path(args.models_dir) / f"{args.variant}_seed{r['seed']}.pt")
        idx = np.asarray(r["test_idx"])
        orig = dict(zip(idx.tolist(), r["test_scores"], strict=True))
        for kind, level in PERTURBATIONS:
            keep = [i for i in idx if out[(names[i], kind, level)] is not None]
            if len(keep) < 20 or len(set(y[keep])) < 2:
                continue
            g, l_, e = (np.stack([out[(names[i], kind, level)][k] for i in keep]) for k in range(3))
            s_new = model.predict_proba(g, l_, e)
            s_old = np.array([orig[i] for i in keep])
            new = evaluate_scores(y[keep], s_new, threshold=thr)
            old = evaluate_scores(y[keep], s_old, threshold=thr)
            rows.append(
                {
                    "seed": r["seed"],
                    "kind": kind,
                    "level": level,
                    "n_test": len(idx),
                    "n_kept": len(keep),
                    **{f"{m}": new[m] for m in ("pr_auc", "roc_auc", "recall", "precision")},
                    **{f"orig_{m}": old[m] for m in ("pr_auc", "roc_auc", "recall", "precision")},
                }
            )
    df = pd.DataFrame(rows)
    agg = {}
    for (kind, level), g in df.groupby(["kind", "level"]):
        entry = {
            "n_seeds": len(g),
            "n_kept_mean": float(g.n_kept.mean()),
            "n_test_mean": float(g.n_test.mean()),
        }
        for m in ("pr_auc", "roc_auc", "recall", "precision"):
            entry[m] = {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1))}
            d = g[m] - g[f"orig_{m}"]
            entry[f"{m}_vs_original"] = {"mean": float(d.mean()), "std": float(d.std(ddof=1))}
        agg[_label(kind, level)] = entry
    print("\nMedia +- std sui seed (stesso sottoinsieme di KOI per originale e perturbato):")
    for k, e in agg.items():
        print(
            f"  {k:14s} kept {e['n_kept_mean']:.0f}/{e['n_test_mean']:.0f} | "
            f"PR-AUC {e['pr_auc']['mean']:.3f} ({e['pr_auc_vs_original']['mean']:+.3f}) | "
            f"recall {e['recall']['mean']:.3f} ({e['recall_vs_original']['mean']:+.3f}) | "
            f"precision {e['precision']['mean']:.3f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = list(agg)
    for ax, m in zip(axes, ("pr_auc", "recall"), strict=True):
        ax.bar(labels, [agg[k][m]["mean"] for k in labels], yerr=[agg[k][m]["std"] for k in labels])
        ax.set(title=m, ylim=(0, 1))
        ax.tick_params(axis="x", rotation=30)
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=110)
    plt.close(fig)
    Path(args.report).write_text(
        json.dumps({"args": vars(args), "summary": agg, "rows": rows}, indent=2)
    )
    print(f"Salvato {args.report} e {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
