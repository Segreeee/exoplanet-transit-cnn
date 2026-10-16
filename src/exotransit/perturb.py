"""Iniezione di transiti sintetici e perturbazioni di curve reali (Fase 4).

Servono a misurare dove il classificatore smette di funzionare: completezza in funzione di
profondita' e periodo (injection-recovery) e degrado con meno dati o piu' rumore (robustezza).
"""

from __future__ import annotations

import numpy as np

from exotransit.preprocess import normalize_flux, robust_sigma, wrapped_phase
from exotransit.synthetic import trapezoid_profile


def transit_duration_days(period: float) -> float:
    """Durata di un transito centrale attorno a una stella tipo Sole: 13 h * (P / 1 anno)^(1/3).

    La durata cresce con P^(1/3) a densita' stellare fissa (l'orbita e' piu' larga e il pianeta
    piu' lento, ma la corda sul disco stellare cresce piano). E' un'approssimazione.
    """
    return 13.0 / 24.0 * (period / 365.25) ** (1.0 / 3.0)


def inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    depth: float,
    duration: float,
    ingress_frac: float = 0.15,
) -> np.ndarray:
    """Moltiplica il flusso per un transito periodico trapezoidale (profondita' = frazione)."""
    profile = trapezoid_profile(wrapped_phase(time, period, t0), duration, ingress_frac)
    return flux * (1.0 - depth * profile)


def point_scatter(flux: np.ndarray) -> float:
    """Scatter punto-a-punto robusto (relativo): MAD delle differenze successive / sqrt(2).

    Insensibile a trend lenti e a pochi outlier; e' la misura di "quanto e' rumorosa" una curva.
    """
    flux = normalize_flux(np.asarray(flux, dtype=float))
    return robust_sigma(np.diff(flux)) / np.sqrt(2.0)


def perturb_light_curve(
    time: np.ndarray, flux: np.ndarray, kind: str, level: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """`truncate`: tiene i primi `level` giorni. `noise`: porta lo scatter punto-a-punto a
    `level` volte l'originale aggiungendo rumore gaussiano (level=1 non cambia nulla)."""
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    if kind == "truncate":
        keep = time < time[0] + level
        return time[keep], flux[keep]
    if kind == "noise":
        if level <= 1.0:
            return time, flux
        extra = np.sqrt(level**2 - 1.0) * point_scatter(flux) * np.nanmedian(flux)
        return time, flux + rng.normal(0.0, extra, flux.size)
    raise ValueError(f"perturbazione sconosciuta: {kind}")
