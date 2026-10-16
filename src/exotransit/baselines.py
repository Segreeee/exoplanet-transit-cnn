"""Baseline di Fase 2: il numero da battere per la CNN.

Due modelli che NON guardano le curve, solo le 4 feature scalari (`extra`):
- `snr_only`: punteggio = SNR (colonna 3, log1p(SNR) del catalogo KOI). Ranking identico a SNR.
- `random_forest_snr_only`: random forest sul solo SNR (cattura la relazione non monotona:
  SNR altissimo = quasi sempre binaria a eclisse, quindi falso positivo).
- `random_forest_scalar`: random forest sulle 4 feature (profondita', periodo, durata/periodo, SNR).

Attenzione: per i KOI le feature vengono dalla tabella NASA (effemeride nota), non da una misura
fatta qui sulla curva: sono il lavoro della pipeline Kepler, quindi la baseline e' forte.
Soglia scelta sulla validazione (F1 massimo), metriche sul test con IC bootstrap per stella.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from exotransit.metrics import (
    best_f1_threshold,
    bootstrap_ci,
    evaluate_scores,
    precision_at_recall,
)
from exotransit.splits import split_by_group

SNR_COLUMN = 3

_RANKING_METRICS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "precision_at_recall90": precision_at_recall,
}


def score_model(y_val, s_val, y_test, s_test, groups_test, n_boot, seed, thr=None) -> dict:
    """Metriche sul test con IC bootstrap per stella; soglia dalla validazione (o `thr` imposta)."""
    test = evaluate_scores(y_test, s_test)
    thr = best_f1_threshold(y_val, s_val) if thr is None else thr
    ci = {
        name: bootstrap_ci(fn, y_test, s_test, groups_test, n_boot=n_boot, seed=seed)
        for name, fn in _RANKING_METRICS.items()
    }
    return {
        "threshold_from_val": thr,
        "test": {
            **{
                k: test[k]
                for k in ("roc_auc", "pr_auc", "precision_at_recall90", "n", "prevalence")
            },
            "ci95": {k: list(v) for k, v in ci.items()},
            "at_val_threshold": evaluate_scores(y_test, s_test, threshold=thr),
        },
    }


def baseline_report(
    extra: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0, n_boot: int = 1000
) -> dict:
    """Split per stella, addestra le baseline sul train e valuta su val/test."""
    extra, y, groups = np.asarray(extra), np.asarray(y), np.asarray(groups)
    if len(np.unique(groups)) < 10:
        raise ValueError("Servono almeno 10 stelle distinte per lo split")
    splits = split_by_group(y, groups, seed=seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]

    def forest(columns):
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
        return rf.fit(extra[tr][:, columns], y[tr]), columns

    rf_snr, snr_cols = forest([SNR_COLUMN])
    rf, all_cols = forest(list(range(extra.shape[1])))
    scores = {
        "snr_only": lambda idx: extra[idx, SNR_COLUMN],
        "random_forest_snr_only": lambda idx: rf_snr.predict_proba(extra[idx][:, snr_cols])[:, 1],
        "random_forest_scalar": lambda idx: rf.predict_proba(extra[idx][:, all_cols])[:, 1],
    }
    return {
        "splits": {
            name: {
                "n": int(len(idx)),
                "n_stars": int(len(np.unique(groups[idx]))),
                "n_positives": int(y[idx].sum()),
            }
            for name, idx in splits.items()
        },
        "models": {
            name: score_model(y[va], f(va), y[te], f(te), groups[te], n_boot, seed)
            for name, f in scores.items()
        },
    }
