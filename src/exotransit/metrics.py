"""Metriche per classificazione sbilanciata. Preferisci PR-AUC all'accuratezza."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def precision_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.9
) -> float:
    """Massima precisione ottenibile con recall >= target_recall (NaN se impossibile)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = recall >= target_recall
    return float(precision[ok].max()) if ok.any() else float("nan")


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """ROC-AUC, PR-AUC, precision@recall=0.9 e matrice di confusione alla soglia data.

    Con una sola classe presente le metriche di ranking sono NaN invece di sollevare errori.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    two_classes = len(np.unique(y_true)) == 2
    pred = y_score >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if two_classes else float("nan"),
        "precision_at_recall90": precision_at_recall(y_true, y_score)
        if two_classes
        else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray, default: float = 0.5) -> float:
    """Soglia che massimizza F1 sui dati dati (da usare sulla VALIDAZIONE, mai sul test).

    Con classi sbilanciate o punteggi poco calibrati 0.5 e' spesso una soglia sbagliata: un
    modello con ottimo ranking (ROC-AUC alta) puo' non classificare nessun positivo a 0.5.
    Senza positivi restituisce `default`. A parita' di F1 sceglie la soglia piu' alta.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return default
    precision, recall, thresholds = precision_recall_curve(y_true, np.asarray(y_score, float))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)[-1]
    return float(thresholds[best])


def bootstrap_ci(
    metric,
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Intervallo bootstrap percentile di `metric(y, score)`, ricampionando STELLE intere.

    Ricampionare per riga sottostimerebbe l'incertezza: i KOI della stessa stella non sono
    indipendenti. Le repliche con una sola classe (metrica indefinita) sono scartate; se
    tutte lo sono restituisce (NaN, NaN).
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric(y_true[idx], y_score[idx])))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
