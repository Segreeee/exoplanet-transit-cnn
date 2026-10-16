"""Baseline di Fase 2 su un dataset .npz: SNR da solo e random forest sulle feature scalari.

    python scripts/baselines.py --data data/processed/kepler.npz \
        --report reports/kepler_baselines.json

Stesso split per stella di `train.py` (stesso seed): le tabelle sono confrontabili con la CNN.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from exotransit.baselines import baseline_report


def _fmt(m: dict, key: str) -> str:
    lo, hi = m["ci95"][key]
    return f"{m[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", default="reports/baselines.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    with np.load(args.data) as d:
        extra, y, groups = d["extra"], d["y"], d["groups"]
    report = {
        "args": vars(args),
        "versions": {p: version(p) for p in ("numpy", "scikit-learn")},
        **baseline_report(extra, y, groups, seed=args.seed, n_boot=args.n_boot),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    t = report["splits"]["test"]
    print(f"Test: {t['n']} esempi, {t['n_stars']} stelle, {t['n_positives']} positivi")
    for name, r in report["models"].items():
        m, c = r["test"], r["test"]["at_val_threshold"]
        print(
            f"{name}: ROC-AUC {_fmt(m, 'roc_auc')} | PR-AUC {_fmt(m, 'pr_auc')} | "
            f"P@R90 {_fmt(m, 'precision_at_recall90')}\n"
            f"   soglia(val) {r['threshold_from_val']:.3f}: precision {c['precision']:.3f} "
            f"recall {c['recall']:.3f} | tp {c['tp']} fp {c['fp']} fn {c['fn']} tn {c['tn']}"
        )
    print(f"Salvato {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
