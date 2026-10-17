"""Controlli indipendenti: il risultato riportato regge? Non si fida del codice di train.py.

    python scripts/sanity_checks.py --data data/processed/synthetic.npz \
        --model models/synthetic_cnn.pt --report reports/synthetic_metrics.json \

Controlli (esce con codice 1 se uno FALLISCE):
 1. Nessuna stella compare in due split (verifica rifatta qui con insiemi, senza assert_no_leakage).
 2. Ricalcolo delle metriche del test con sklearn dal modello salvato e confronto col report JSON.
 3. (rimosso: la coerenza della demo web e' in scripts/check_demo.py)
 4. Label shuffle: addestrando con etichette di training mescolate, il test deve crollare al
    caso (PR-AUC ~ prevalenza). Se resta alto c'e' leakage o un bug nella valutazione.
 5. Informativo: AUC della sola feature SNR (se gia' alta, il compito e' facile da scorciatoia).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from exotransit.model import TwoTowerCNN, fit
from exotransit.splits import split_by_group

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{RESULTS[-1][1]}] {name}: {detail}", flush=True)


def info(name: str, detail: str) -> None:
    RESULTS.append((name, "INFO", detail))
    print(f"[INFO] {name}: {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--skip-shuffle", action="store_true", help="salta il controllo piu' lento")
    ap.add_argument("--shuffle-epochs", type=int, default=15)
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    seed = report["args"]["seed"]
    with np.load(args.data, allow_pickle=False) as d:
        g, loc, ext, y, groups = (
            d[k] for k in ("global_view", "local_view", "extra", "y", "groups")
        )
    splits = split_by_group(y, groups, seed=seed)

    # 1. leakage, rifatto con insiemi
    sets = {k: set(groups[v].tolist()) for k, v in splits.items()}
    overlaps = {
        f"{a}&{b}": len(sets[a] & sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    record(
        "nessuna stella in due split",
        all(v == 0 for v in overlaps.values()),
        f"sovrapposizioni {overlaps}; stelle per split { {k: len(v) for k, v in sets.items()} }",
    )
    covered = sum(len(v) for v in splits.values())
    record("tutti gli esempi assegnati una volta", covered == len(y), f"{covered}/{len(y)}")

    # 2. ricalcolo metriche dal modello salvato
    ckpt = torch.load(args.model, map_location="cpu")
    model = TwoTowerCNN(base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    te = splits["test"]
    scores = model.predict_proba(g[te], loc[te], ext[te])
    roc, pr = roc_auc_score(y[te], scores), average_precision_score(y[te], scores)
    rep = report["test_cnn"]
    record(
        "metriche del report = ricalcolo indipendente",
        abs(roc - rep["roc_auc"]) < 1e-6 and abs(pr - rep["pr_auc"]) < 1e-6,
        f"ROC-AUC {roc:.4f} (report {rep['roc_auc']:.4f}), "
        f"PR-AUC {pr:.4f} (report {rep['pr_auc']:.4f})",
    )
    thr = float(ckpt["threshold"])
    pred = scores >= thr
    tp = int(np.sum(pred & (y[te] == 1)))
    fp = int(np.sum(pred & (y[te] == 0)))
    tuned = report["test_cnn_at_val_threshold"]
    record(
        "matrice di confusione alla soglia = report",
        (tp, fp) == (tuned["tp"], tuned["fp"])
        and abs(thr - report["threshold_cnn_from_val"]) < 1e-9,
        f"tp {tp} fp {fp} (report {tuned['tp']}, {tuned['fp']}), soglia {thr:.4f}",
    )

    # 4. label shuffle
    if not args.skip_shuffle:
        tr, va = splits["train"], splits["val"]
        rng = np.random.default_rng(seed + 1)
        y_shuf = rng.permutation(y[tr])
        torch.manual_seed(seed)
        m2 = TwoTowerCNN(base=ckpt["base"])
        fit(
            m2,
            (g[tr], loc[tr], ext[tr], y_shuf),
            (g[va], loc[va], ext[va], y[va]),
            epochs=args.shuffle_epochs,
            seed=seed,
        )
        s2 = m2.predict_proba(g[te], loc[te], ext[te])
        pr2 = average_precision_score(y[te], s2)
        prev = float(y[te].mean())
        record(
            "label shuffle: la CNN crolla al caso",
            pr2 < prev + 0.2,
            f"PR-AUC con etichette mescolate {pr2:.3f} "
            f"(caso ~ {prev:.2f}; soglia {prev + 0.2:.2f})",
        )
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
        rf.fit(ext[tr], y_shuf)
        pr3 = average_precision_score(y[te], rf.predict_proba(ext[te])[:, 1])
        record(
            "label shuffle: la baseline crolla al caso",
            pr3 < prev + 0.2,
            f"PR-AUC {pr3:.3f} (caso ~ {prev:.2f})",
        )

    # 5. scorciatoie: AUC di ogni singola feature scalare sul test
    names = ["log10 profondita'", "log10 periodo", "durata/periodo", "log1p SNR"]
    aucs = {n: roc_auc_score(y[te], ext[te, i]) for i, n in enumerate(names)}
    info(
        "ROC-AUC di una singola feature (0.5 = inutile, 1 o 0 = scorciatoia)",
        ", ".join(f"{n}: {a:.2f}" for n, a in aucs.items()),
    )

    failed = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(failed)} controlli falliti su {sum(r[1] != 'INFO' for r in RESULTS)}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
