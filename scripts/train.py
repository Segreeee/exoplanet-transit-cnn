"""Addestra la CNN a due torri su un dataset .npz e salva modello + metriche.

    python scripts/train.py --data data/processed/synthetic.npz --epochs 30 \
        --out models/synthetic_cnn.pt --report reports/synthetic_metrics.json

Calcola anche una baseline senza CNN (random forest sulle sole feature scalari): se la CNN
non la batte nettamente, le viste non stanno aggiungendo informazione.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from exotransit.metrics import best_f1_threshold, evaluate_scores
from exotransit.model import TwoTowerCNN, count_parameters, fit, pick_device
from exotransit.splits import split_by_group


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/cnn.pt")
    ap.add_argument("--report", default="reports/metrics.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=16, help="canali della prima convoluzione")
    args = ap.parse_args()

    with np.load(args.data) as d:
        g, l_, e, y, groups = (d[k] for k in ("global_view", "local_view", "extra", "y", "groups"))
    splits = split_by_group(y, groups, seed=args.seed)  # SEMPRE per stella
    take = lambda idx: (g[idx], l_[idx], e[idx], y[idx])  # noqa: E731
    train, val, test = (take(splits[k]) for k in ("train", "val", "test"))

    device = pick_device()
    print(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )
    print({k: f"{len(v)} esempi, {int(y[v].sum())} positivi" for k, v in splits.items()})

    torch.manual_seed(args.seed)
    model = TwoTowerCNN(base=args.base)
    print(f"Parametri: {count_parameters(model):,}")
    res = fit(
        model,
        train,
        val,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    cnn_scores = model.predict_proba(test[0], test[1], test[2])
    cnn = evaluate_scores(test[3], cnn_scores)  # soglia 0.5, come riferimento
    # La soglia si sceglie sulla VALIDAZIONE (mai sul test): i punteggi non sono calibrati.
    cnn_thr = best_f1_threshold(val[3], model.predict_proba(val[0], val[1], val[2]))
    cnn_tuned = evaluate_scores(test[3], cnn_scores, threshold=cnn_thr)

    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, class_weight="balanced")
    rf.fit(train[2], train[3])
    rf_test = rf.predict_proba(test[2])[:, 1]
    baseline = evaluate_scores(test[3], rf_test)
    rf_thr = best_f1_threshold(val[3], rf.predict_proba(val[2])[:, 1])
    baseline_tuned = evaluate_scores(test[3], rf_test, threshold=rf_thr)

    report = {
        "args": vars(args),
        "device": str(device),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "best_epoch": res.best_epoch,
        "val_pr_auc_best": res.best_val_pr_auc,
        "test_cnn": cnn,
        "test_baseline_random_forest_on_scalar_features": baseline,
        "threshold_cnn_from_val": cnn_thr,
        "threshold_rf_from_val": rf_thr,
        "test_cnn_at_val_threshold": cnn_tuned,
        "test_baseline_rf_at_val_threshold": baseline_tuned,
        "history": res.history,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "base": args.base, "threshold": cnn_thr}, args.out
    )

    for name, m in (("CNN", cnn), ("Baseline RF (solo feature scalari)", baseline)):
        print(
            f"{name}: ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | "
            f"P@R90 {m['precision_at_recall90']:.3f} | prevalenza {m['prevalence']:.2f}"
        )
    for name, m, thr in (
        ("CNN", cnn_tuned, cnn_thr),
        ("Baseline RF", baseline_tuned, rf_thr),
    ):
        print(
            f"{name} @soglia-da-validazione {thr:.3f}: precision {m['precision']:.3f} | "
            f"recall {m['recall']:.3f} | tp {m['tp']} fp {m['fp']} fn {m['fn']} tn {m['tn']}"
        )
    print(f"Salvati {args.out} e {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
