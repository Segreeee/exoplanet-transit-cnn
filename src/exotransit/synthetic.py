"""Curve di luce sintetiche per sviluppo, test e injection-recovery.

Servono a verificare la pipeline senza scaricare nulla. NON sostituiscono i dati
reali: i risultati su sintetici dicono che il codice funziona, non che il modello
sia scientificamente valido.

Classi generate:
- "planet": transito periodico a forma di trapezio (label 1)
- "eb":     binaria a eclisse: transito profondo/a V, eclisse secondaria a fase 0.5
            e differenza tra transiti pari e dispari (label 0)
- "noise":  nessun transito periodico, solo variabilita', rumore e dip isolati (label 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exotransit.preprocess import wrapped_phase

KINDS = ("planet", "eb", "noise")


@dataclass
class SyntheticExample:
    time: np.ndarray
    flux: np.ndarray
    kind: str
    label: int
    period: float  # NaN per "noise"
    t0: float  # NaN per "noise"
    duration: float  # giorni; NaN per "noise"
    depth: float  # frazione del flusso; NaN per "noise"
    star_id: int


def trapezoid_profile(phase_days: np.ndarray, duration: float, ingress_frac: float = 0.15):
    """Profilo del transito tra 0 (fuori) e 1 (fondo piatto).

    `duration` e' la durata totale (primo-ultimo contatto). `ingress_frac` e' la
    frazione di durata occupata dall'ingresso (e dall'uscita); 0.5 da' un transito a V.
    """
    if not 0.0 < ingress_frac <= 0.5:
        raise ValueError("ingress_frac deve stare in (0, 0.5]")
    half = duration / 2.0
    width = ingress_frac * duration
    return np.clip((half - np.abs(phase_days)) / width, 0.0, 1.0)


def stellar_variability(time: np.ndarray, rng: np.random.Generator, amplitude: float):
    """Variabilita' stellare lenta: somma di 1-3 sinusoidi con periodi di 2-15 giorni."""
    flux = np.zeros_like(time)
    for _ in range(int(rng.integers(1, 4))):
        period = rng.uniform(2.0, 15.0)
        amp = rng.uniform(0.2, 1.0) * amplitude
        flux += amp * np.sin(2 * np.pi * time / period + rng.uniform(0, 2 * np.pi))
    return flux


def _drop_gaps(time, flux, rng, n_gaps):
    keep = np.ones(time.size, dtype=bool)
    for _ in range(n_gaps):
        start = rng.uniform(time[0], time[-1])
        keep &= ~((time >= start) & (time < start + rng.uniform(0.5, 2.0)))
    return time[keep], flux[keep]


def make_example(
    rng: np.random.Generator,
    kind: str | None = None,
    star_id: int = 0,
    baseline_days: float = 90.0,
    cadence_min: float = 30.0,
) -> SyntheticExample:
    """Genera una curva di luce sintetica con segnale di tipo `kind`."""
    if kind is None:
        kind = str(rng.choice(KINDS))
    if kind not in KINDS:
        raise ValueError(f"kind sconosciuto: {kind!r}")

    time = np.arange(0.0, baseline_days, cadence_min / 1440.0)
    sigma = 10 ** rng.uniform(-3.7, -3.0)  # ~200-1000 ppm per punto
    flux = (
        1.0
        + stellar_variability(time, rng, amplitude=10 ** rng.uniform(-3.5, -2.3))
        + rng.normal(0.0, sigma, time.size)
    )

    period = t0 = duration = depth = float("nan")

    if kind in ("planet", "eb"):
        period = float(rng.uniform(0.8, baseline_days / 3.0))
        t0 = float(rng.uniform(0.0, period))
        # ordine di grandezza della durata per orbita circolare attorno a una stella solare
        duration = float(0.54 * (period / 365.0) ** (1 / 3) * rng.uniform(0.5, 1.2))
        phase = wrapped_phase(time, period, t0)

        if kind == "planet":
            depth = float(min(sigma * 10 ** rng.uniform(0.2, 1.5), 0.02))
            flux -= depth * trapezoid_profile(phase, duration, ingress_frac=rng.uniform(0.05, 0.25))
        else:
            deep = rng.random() < 0.5
            depth = (
                float(10 ** rng.uniform(-2.3, -0.8))
                if deep
                else float(min(sigma * 10 ** rng.uniform(0.2, 1.5), 0.02))
            )
            ingress = rng.uniform(0.25, 0.5)
            transit_index = np.round((time - t0) / period).astype(int)
            even_scale = np.where(transit_index % 2 == 0, rng.uniform(0.8, 1.0), 1.0)
            flux -= depth * even_scale * trapezoid_profile(phase, duration, ingress)
            secondary_phase = wrapped_phase(time, period, t0 + 0.5 * period)
            flux -= (
                depth
                * rng.uniform(0.3, 1.0)
                * trapezoid_profile(secondary_phase, duration, ingress)
            )
    else:
        for _ in range(int(rng.integers(1, 4))):
            center = rng.uniform(time[0], time[-1])
            width = rng.uniform(0.05, 0.3)
            amp = sigma * 10 ** rng.uniform(0.2, 1.2)
            flux -= amp * np.exp(-0.5 * ((time - center) / width) ** 2)

    time, flux = _drop_gaps(time, flux, rng, n_gaps=int(rng.integers(0, 4)))
    return SyntheticExample(
        time=time,
        flux=flux,
        kind=kind,
        label=int(kind == "planet"),
        period=period,
        t0=t0,
        duration=duration,
        depth=depth,
        star_id=star_id,
    )


def make_dataset(
    n: int,
    seed: int = 0,
    examples_per_star: int = 2,
    class_probs: tuple[float, float, float] = (0.5, 0.25, 0.25),
    **kwargs,
) -> list[SyntheticExample]:
    """Genera `n` esempi. Piu' esempi condividono la stessa stella (`star_id`), come i TCE
    reali: servono per verificare che gli split per stella evitino il data leakage."""
    if examples_per_star < 1:
        raise ValueError("examples_per_star deve essere >= 1")
    rng = np.random.default_rng(seed)
    kinds = rng.choice(KINDS, size=n, p=class_probs)
    return [
        make_example(rng, kind=str(k), star_id=i // examples_per_star, **kwargs)
        for i, k in enumerate(kinds)
    ]
