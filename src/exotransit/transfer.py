"""Trasferimento Kepler -> TESS (Fase 5).

Quattro approcci, tutti valutati sullo stesso test TESS (split per stella, uno per seed):
- `kepler_as_is`: il modello Kepler `no_scalar` del seed, senza riaddestrare (soglia di Kepler);
- `fine_tuned`: lo stesso modello riaddestrato sul train TESS (lr basso, poche epoche);
- `tess_from_scratch`: stessa architettura addestrata solo su TESS;
- `random_forest_3_features`: profondita', periodo, durata/periodo (il catalogo TOI non ha SNR).
Piu' `kepler_as_is_all_tess`: il modello Kepler su TUTTI i TESS con IC bootstrap per stella.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from exotransit.baselines import score_model
from exotransit.experiments import METRICS, load_model
from exotransit.model import TwoTowerCNN, fit, pick_device
from exotransit.splits import split_by_group

APPROACHES = ("kepler_as_is", "fine_tuned", "tess_from_scratch", "random_forest_3_features")
_REFERENCES = ("kepler_as_is", "tess_from_scratch", "random_forest_3_features")


def run_transfer(
    g,
    l_,
    extra,
    y,
    groups,
    models_dir,
    seeds,
    variant: str = "no_scalar",
    epochs_finetune: int = 10,
    lr_finetune: float = 1e-4,
    epochs_scratch: int = 30,
    lr_scratch: float = 1e-3,
    batch_size: int = 64,
    n_boot: int = 1000,
    log=None,
) -> dict:
    device = pick_device()
    runs, whole = [], {}
    for seed in seeds:
        kep_model, kep_thr = load_model(Path(models_dir) / f"{variant}_seed{seed}.pt")
        s_all = kep_model.predict_proba(g, l_, extra)
        whole[seed] = score_model(y, s_all, y, s_all, groups, n_boot, seed, thr=kep_thr)

        splits = split_by_group(y, groups, seed=seed)  # SEMPRE per stella
        tr, va, te = splits["train"], splits["val"], splits["test"]
        train, val = (tuple(a[i] for a in (g, l_, extra, y)) for i in (tr, va))
        score = lambda m, idx: m.predict_proba(g[idx], l_[idx], extra[idx])  # noqa: E731
        scores = {"kepler_as_is": (score(kep_model, va), score(kep_model, te), kep_thr)}

        torch.manual_seed(seed)
        ft, _ = load_model(Path(models_dir) / f"{variant}_seed{seed}.pt")
        fit(
            ft,
            train,
            val,
            epochs=epochs_finetune,
            lr=lr_finetune,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        scores["fine_tuned"] = (score(ft, va), score(ft, te), None)

        torch.manual_seed(seed)
        scratch = TwoTowerCNN(n_extra=0, base=16)
        fit(
            scratch,
            train,
            val,
            epochs=epochs_scratch,
            lr=lr_scratch,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        scores["tess_from_scratch"] = (score(scratch, va), score(scratch, te), None)

        rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
        rf.fit(extra[tr][:, :3], y[tr])
        scores["random_forest_3_features"] = (
            rf.predict_proba(extra[va][:, :3])[:, 1],
            rf.predict_proba(extra[te][:, :3])[:, 1],
            None,
        )
        approaches = {
            name: score_model(y[va], sv, y[te], st, groups[te], n_boot, seed, thr)
            for name, (sv, st, thr) in scores.items()
        }
        runs.append(
            {
                "seed": int(seed),
                "split": {
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "n_test_stars": int(len(np.unique(groups[te]))),
                },
                "approaches": approaches,
            }
        )
        if log:
            log(
                f"seed {seed}: "
                + " | ".join(
                    f"{a} PR-AUC {approaches[a]['test']['pr_auc']:.3f}" for a in APPROACHES
                )
            )
    return {"runs": runs, "kepler_as_is_all_tess": whole}


def summarize_transfer(runs: list[dict]) -> dict:
    """Media e std (ddof=1) sui seed per approccio e differenze appaiate con gli altri approcci."""
    std = lambda a: float(a.std(ddof=1)) if len(a) > 1 else float("nan")  # noqa: E731
    out = {}
    for name in APPROACHES:
        entry = {"n_seeds": len(runs), "metrics": {}, "paired_diff": {}}
        for m in METRICS:
            v = np.array([r["approaches"][name]["test"][m] for r in runs], dtype=float)
            entry["metrics"][m] = {"mean": float(v.mean()), "std": std(v), "values": v.tolist()}
        for ref in (r for r in _REFERENCES if r != name):
            entry["paired_diff"][f"vs_{ref}"] = {}
            for m in METRICS:
                d = np.array(
                    [
                        r["approaches"][name]["test"][m] - r["approaches"][ref]["test"][m]
                        for r in runs
                    ]
                )
                entry["paired_diff"][f"vs_{ref}"][m] = {
                    "mean": float(d.mean()),
                    "std": std(d),
                    "n_seeds_better": int((d > 0).sum()),
                }
        out[name] = entry
    return out
