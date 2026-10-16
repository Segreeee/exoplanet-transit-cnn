"""Esperimenti di Fase 3: CNN e ablation su piu' seed, appaiate alla baseline random forest.

Per ogni seed: split per stella diverso (`split_by_group(seed=...)`); su QUELLO split si addestrano
le varianti della CNN e la baseline random forest, cosi' la differenza CNN - RF e' appaiata. Si
riportano media e deviazione standard sui seed, non il seed migliore. Soglia dalla validazione.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from exotransit.baselines import baseline_report
from exotransit.metrics import best_f1_threshold, evaluate_scores
from exotransit.model import TwoTowerCNN, count_parameters, fit, pick_device
from exotransit.preprocess import N_EXTRA
from exotransit.splits import split_by_group

# kwargs di TwoTowerCNN per ogni ablation (le viste restano nei dati, il modello le ignora)
VARIANTS: dict[str, dict] = {
    "full": {},
    "global_only": {"use_local": False},
    "local_only": {"use_global": False},
    "no_scalar": {"n_extra": 0},
}
METRICS = ("roc_auc", "pr_auc", "precision_at_recall90")


def shuffle_labels_by_group(y, groups, seed: int = 0):
    """Controllo anti-leakage: permuta le etichette tra STELLE (ogni stella tiene un'etichetta
    sola, quella massima dei suoi esempi). Un modello addestrato cosi' deve fare ~0.5 di ROC-AUC."""
    y, groups = np.asarray(y), np.asarray(groups)
    stars, inverse = np.unique(groups, return_inverse=True)
    star_label = np.array([y[groups == s].max() for s in stars])
    return np.random.default_rng(seed).permutation(star_label)[inverse]


def load_model(path):
    """(modello in eval, soglia dalla validazione) da un file salvato da `run_experiments`."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TwoTowerCNN(**ck["kwargs"])
    model.load_state_dict(ck["state_dict"])
    return model.to(pick_device()).eval(), ck["threshold"]


def run_experiments(
    g,
    l_,
    extra,
    y,
    groups,
    seeds,
    variants=tuple(VARIANTS),
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    base: int = 16,
    log=None,
    models_dir=None,
) -> dict:
    device = pick_device()
    runs, rf_by_seed = [], {}
    for seed in seeds:
        splits = split_by_group(y, groups, seed=seed)  # SEMPRE per stella
        train, val, test = (
            tuple(a[splits[k]] for a in (g, l_, extra, y)) for k in ("train", "val", "test")
        )
        rf = baseline_report(extra, y, groups, seed=seed, n_boot=0)["models"]
        rf_by_seed[seed] = rf["random_forest_scalar"]["test"]
        for name in variants:
            torch.manual_seed(seed)
            kwargs = {"n_extra": N_EXTRA, "base": base, **VARIANTS[name]}
            model = TwoTowerCNN(**kwargs)
            res = fit(
                model,
                train,
                val,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )
            s_test = model.predict_proba(test[0], test[1], test[2])
            s_val = model.predict_proba(val[0], val[1], val[2])
            thr = best_f1_threshold(val[3], s_val)
            ev = evaluate_scores(test[3], s_test)
            runs.append(
                {
                    "variant": name,
                    "seed": int(seed),
                    "n_params": count_parameters(model),
                    "best_epoch": res.best_epoch,
                    "val_pr_auc_best": res.best_val_pr_auc,
                    "test": {k: ev[k] for k in (*METRICS, "n", "prevalence")},
                    "threshold_from_val": thr,
                    "at_val_threshold": evaluate_scores(test[3], s_test, threshold=thr),
                    "history": res.history,
                    "val_idx": splits["val"].tolist(),
                    "val_scores": s_val.tolist(),
                    "test_idx": splits["test"].tolist(),
                    "test_scores": s_test.tolist(),
                }
            )
            if models_dir is not None:
                Path(models_dir).mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"state_dict": model.state_dict(), "kwargs": kwargs, "threshold": thr},
                    Path(models_dir) / f"{name}_seed{seed}.pt",
                )
            if log:
                log(f"seed {seed} {name}: PR-AUC {ev['pr_auc']:.3f} ROC-AUC {ev['roc_auc']:.3f}")
    return {"runs": runs, "baseline_random_forest_by_seed": rf_by_seed}


def summarize_runs(runs: list[dict], rf_by_seed: dict) -> dict:
    """Media e std (ddof=1) sui seed per variante, e differenza appaiata CNN - random forest."""
    out: dict = {}
    for name in dict.fromkeys(r["variant"] for r in runs):
        rs = sorted((r for r in runs if r["variant"] == name), key=lambda r: r["seed"])
        entry = {"n_seeds": len(rs), "metrics": {}, "paired_diff_vs_rf": {}}
        for m in METRICS:
            v = np.array([r["test"][m] for r in rs], dtype=float)
            d = np.array([r["test"][m] - rf_by_seed[r["seed"]][m] for r in rs], dtype=float)
            std = lambda a: float(a.std(ddof=1)) if len(a) > 1 else float("nan")  # noqa: E731
            entry["metrics"][m] = {"mean": float(v.mean()), "std": std(v), "values": v.tolist()}
            entry["paired_diff_vs_rf"][m] = {
                "mean": float(d.mean()),
                "std": std(d),
                "n_seeds_better": int((d > 0).sum()),
            }
        out[name] = entry
    return out
