"""Dati per la demo web e per le schede di vetting: selezione degli esempi, finestra della curva
da mostrare e fatti calibrati. Funzioni pure (nessun I/O) per poterle testare."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from exotransit.llm import candidate_facts
from exotransit.preprocess import ephemeris_from_extra


def select_examples(y, scores, thr: float, idx, n_each: dict, seed: int = 0) -> list[dict]:
    """Sceglie a caso (seed fisso) fino a `n_each[esito]` esempi per ognuno di TP/TN/FP/FN
    tra gli indici `idx`. Gli errori sono sovra-rappresentati di proposito: la lista NON e'
    un campione rappresentativo, le metriche vengono dal test set intero."""
    y, scores, idx = np.asarray(y), np.asarray(scores, dtype=float), np.asarray(idx)
    rng = np.random.default_rng(seed)
    pred = scores[idx] >= thr
    truth = y[idx] == 1
    groups = {"TP": truth & pred, "TN": ~truth & ~pred, "FP": ~truth & pred, "FN": truth & ~pred}
    out = []
    for outcome, mask in groups.items():
        pool = idx[mask]
        take = min(n_each.get(outcome, 0), len(pool))
        for i in rng.choice(pool, size=take, replace=False) if take else []:
            out.append({"index": int(i), "outcome": outcome})
    return sorted(out, key=lambda p: p["index"])


def transit_window(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    min_days: float = 8.0,
    max_days: float = 80.0,
    n_epochs: int = 300,
) -> tuple[float, float, list[float]]:
    """Finestra di tempo da mostrare: ~4 periodi (tra `min_days` e `max_days`), centrata
    sull'epoca di transito dove ci sono piu' dati. Restituisce (inizio, fine, tempi dei transiti)
    """
    time = np.asarray(time, dtype=float)
    width = float(np.clip(4.0 * period, min_days, max_days))
    k_lo, k_hi = int(np.ceil((time.min() - t0) / period)), int(np.floor((time.max() - t0) / period))
    ks = np.arange(k_lo, k_hi + 1)
    if len(ks) > n_epochs:
        ks = ks[np.linspace(0, len(ks) - 1, n_epochs).astype(int)]
    centers = t0 + ks * period
    counts = np.searchsorted(time, centers + width / 2) - np.searchsorted(time, centers - width / 2)
    center = float(centers[int(np.argmax(counts))]) if len(centers) else float(time.mean())
    start, end = center - width / 2, center + width / 2
    n_lo, n_hi = int(np.ceil((start - t0) / period)), int(np.floor((end - t0) / period))
    return start, end, [round(t0 + n * period, 4) for n in range(n_lo, n_hi + 1)]


def bin_window(time, flux, start: float, end: float, max_points: int = 1200) -> dict:
    """Punti della finestra [start, end]; oltre `max_points` si media per intervalli."""
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    sel = (time >= start) & (time <= end)
    t, f = time[sel], flux[sel]
    if len(t) > max_points:
        edges = np.linspace(start, end, max_points + 1)
        b = np.clip(np.digitize(t, edges) - 1, 0, max_points - 1)
        counts = np.bincount(b, minlength=max_points)
        keep = counts > 0
        t = (np.bincount(b, weights=t, minlength=max_points)[keep] / counts[keep]).copy()
        f = np.bincount(b, weights=f, minlength=max_points)[keep] / counts[keep]
    return {"t": np.round(t, 3).tolist(), "f": np.round(f, 5).tolist()}


def calibrated_facts(names, extra, scores, val_scores, val_y, thr: float) -> list[dict]:
    """Fatti dei candidati con probabilita' CALIBRATA (isotonica sulla validazione).

    L'esito (candidato/scartato) viene dal punteggio grezzo e dalla sua soglia di validazione:
    la calibrazione serve al testo. L'isotonica da' valori a gradini (anche 0.0 e 1.0)."""
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(val_scores), np.asarray(val_y))
    iso_thr = round(float(iso.predict([thr])[0]), 3)
    out = []
    for name, ex, raw in zip(names, extra, np.asarray(scores, dtype=float), strict=True):
        facts = candidate_facts(
            str(name), ephemeris_from_extra(ex), float(iso.predict([raw])[0]), threshold=iso_thr
        )
        facts["esito_classificatore"] = "candidato" if raw >= thr else "scartato"
        out.append(facts)
    return out
