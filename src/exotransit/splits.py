"""Split train/val/test PER STELLA.

Piu' candidati della stessa stella (o la stessa curva con effemeridi diverse) hanno
rumore e sistematiche in comune: se finiscono in split diversi il modello li "riconosce"
e le metriche risultano gonfiate (data leakage). Qui si divide sempre per gruppo.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def split_by_group(
    y: np.ndarray, groups: np.ndarray, seed: int = 0, n_folds: int = 10
) -> dict[str, np.ndarray]:
    """Restituisce indici {"train","val","test"}: ~10% test, ~10% val, il resto train.

    Gruppi disgiunti tra gli split e proporzione delle classi approssimativamente mantenuta.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("y e groups devono avere la stessa lunghezza")
    if len(np.unique(groups)) < n_folds:
        raise ValueError("Servono almeno n_folds gruppi distinti")
    folds = list(
        StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, groups
        )
    )
    test, val = folds[0][1], folds[1][1]
    train = np.concatenate([folds[i][1] for i in range(2, n_folds)])
    splits = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    assert_no_leakage(splits, groups)
    return splits


def assert_no_leakage(splits: dict[str, np.ndarray], groups: np.ndarray) -> None:
    """Solleva AssertionError se un gruppo compare in piu' di uno split."""
    groups = np.asarray(groups)
    names = list(splits)
    sets = {n: set(groups[splits[n]].tolist()) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sets[a] & sets[b]
            assert not shared, f"Leakage: {len(shared)} gruppi condivisi tra {a} e {b}"
