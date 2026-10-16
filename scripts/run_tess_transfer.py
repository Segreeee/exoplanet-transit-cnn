"""Fase 5: trasferimento Kepler -> TESS (as is, fine-tuning, da zero, random forest), 5 seed.

    python scripts/run_tess_transfer.py --data data/processed/tess.npz \
        --models-dir models/kepler --report reports/tess_transfer.json

Per ogni seed: split per stella dei TESS; modello Kepler `no_scalar` dello stesso seed.
`--shuffle-labels`: controllo anti-leakage (etichette TESS permutate tra stelle, ROC-AUC ~0.5).
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from exotransit.experiments import shuffle_labels_by_group
from exotransit.transfer import APPROACHES, run_transfer, summarize_transfer


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--models-dir", default="models/kepler")
    ap.add_argument("--report", default="reports/tess_transfer.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--epochs-scratch", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--shuffle-labels", action="store_true")
    args = ap.parse_args()

    with np.load(args.data) as d:
        g, l_, extra, y, groups = (
            d[k] for k in ("global_view", "local_view", "extra", "y", "groups")
        )
    if args.shuffle_labels:
        y = shuffle_labels_by_group(y, groups, seed=0)
    out = run_transfer(
        g,
        l_,
        extra,
        y,
        groups,
        models_dir=args.models_dir,
        seeds=args.seeds,
        epochs_finetune=args.epochs_finetune,
        lr_finetune=args.lr_finetune,
        epochs_scratch=args.epochs_scratch,
        n_boot=args.n_boot,
        log=lambda s: print(s, flush=True),
    )
    summary = summarize_transfer(out["runs"])
    report = {
        "args": vars(args),
        "versions": {"torch": torch.__version__, "numpy": version("numpy")},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_examples": int(len(y)),
        "n_stars": int(len(np.unique(groups))),
        "n_positives": int(y.sum()),
        "summary": summary,
        **out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print(
        f"\nTESS: {len(y)} esempi, {report['n_stars']} stelle, {int(y.sum())} positivi; "
        f"{len(args.seeds)} seed, media +- std"
    )
    for name in APPROACHES:
        m = summary[name]["metrics"]
        print(f"{name:26s} " + " | ".join(f"{k} {m[k]['mean']:.3f}+-{m[k]['std']:.3f}" for k in m))
    ft = summary["fine_tuned"]["paired_diff"]
    for ref, d in ft.items():
        x = d["pr_auc"]
        print(
            f"fine_tuned PR-AUC {ref}: {x['mean']:+.3f} +- {x['std']:.3f} "
            f"({x['n_seeds_better']}/{len(args.seeds)} seed)"
        )
    w = [v["test"] for v in out["kepler_as_is_all_tess"].values()]
    print(
        "Kepler as-is su TUTTO TESS (soglia Kepler): PR-AUC "
        f"{np.mean([x['pr_auc'] for x in w]):.3f}, ROC-AUC {np.mean([x['roc_auc'] for x in w]):.3f}"
    )
    print(f"Salvato {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
