"""Fase 3: CNN e ablation su piu' seed (split per stella per seed), con baseline appaiata.

    python scripts/run_cnn_experiments.py --data data/processed/kepler.npz \
        --report reports/kepler_cnn.json --seeds 0 1 2 3 4

Varianti: full, global_only, local_only, no_scalar. Iperparametri fissi (nessun tuning).
Salva anche le curve di apprendimento (reports/figures/cnn_learning_curves.png).
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np

from exotransit.experiments import (
    VARIANTS,
    run_experiments,
    shuffle_labels_by_group,
    summarize_runs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_curves(runs: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in (r for r in runs if r["variant"] == "full"):
        h = r["history"]
        axes[0].plot([e["train_loss"] for e in h], color="C0", alpha=0.6)
        axes[0].plot([e["val_loss"] for e in h], color="C1", alpha=0.6)
        axes[1].plot([e["pr_auc"] for e in h], alpha=0.8, label=f"seed {r['seed']}")
    axes[0].set(title="variante full: loss (blu train, arancio val)", xlabel="epoca")
    axes[1].set(title="PR-AUC di validazione", xlabel="epoca")
    axes[1].legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", default="reports/cnn_experiments.json")
    ap.add_argument("--figure", default="reports/figures/cnn_learning_curves.png")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument(
        "--shuffle-labels", action="store_true", help="controllo: etichette permutate tra stelle"
    )
    args = ap.parse_args()

    with np.load(args.data) as d:
        g, l_, extra, y, groups = (
            d[k] for k in ("global_view", "local_view", "extra", "y", "groups")
        )
    if args.shuffle_labels:
        y = shuffle_labels_by_group(y, groups, seed=0)
    out = run_experiments(
        g,
        l_,
        extra,
        y,
        groups,
        seeds=args.seeds,
        variants=args.variants,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        base=args.base,
        log=lambda s: print(s, flush=True),
    )
    summary = summarize_runs(out["runs"], out["baseline_random_forest_by_seed"])
    rf_vals = {
        m: np.array([v[m] for v in out["baseline_random_forest_by_seed"].values()])
        for m in ("roc_auc", "pr_auc", "precision_at_recall90")
    }
    report = {
        "args": vars(args),
        "versions": {"torch": torch.__version__, "numpy": version("numpy")},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_examples": int(len(y)),
        "summary": summary,
        "baseline_random_forest_scalar": {
            m: {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": v.tolist()}
            for m, v in rf_vals.items()
        },
        **out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    plot_curves(out["runs"], Path(args.figure))

    print(f"\n{len(args.seeds)} seed, media +- std (test, per seed split diverso)")
    for name, s in summary.items():
        cells = " | ".join(
            f"{m} {s['metrics'][m]['mean']:.3f}+-{s['metrics'][m]['std']:.3f}"
            for m in ("roc_auc", "pr_auc", "precision_at_recall90")
        )
        d = s["paired_diff_vs_rf"]["pr_auc"]
        better = f"{d['n_seeds_better']}/{s['n_seeds']} seed"
        print(f"{name:12s} {cells} | PR-AUC - RF: {d['mean']:+.3f} ({better})")
    cells = " | ".join(f"{m} {v.mean():.3f}+-{v.std(ddof=1):.3f}" for m, v in rf_vals.items())
    print(f"{'RF scalari':12s} {cells}")
    print(f"Salvati {args.report} e {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
