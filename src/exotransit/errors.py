"""Analisi degli errori per fascia (Fase 4): dove il classificatore sbaglia."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binned_rates(df: pd.DataFrame, column: str, edges) -> pd.DataFrame:
    """Per ogni fascia di `column`: n, n_pos, recall, precisione e tasso di falsi positivi.

    `df` ha le colonne `label` (0/1) e `pred` (bool, alla soglia scelta sulla validazione).
    Le frazioni con denominatore nullo sono NaN, mai errori. Le fasce sono (a, b].
    """
    rows = []
    bins = pd.cut(df[column], list(edges), include_lowest=True)
    for interval, g in df.groupby(bins, observed=False):
        pos, pred = g["label"] == 1, g["pred"].astype(bool)
        tp, fp = int((pos & pred).sum()), int((~pos & pred).sum())
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        rows.append(
            {
                "bin": str(interval),
                "n": len(g),
                "n_pos": n_pos,
                "recall": tp / n_pos if n_pos else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "fpr": fp / n_neg if n_neg else np.nan,
            }
        )
    return pd.DataFrame(rows)
