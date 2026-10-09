"""Preprocessing: detrending, ricerca BLS, ripiegamento in fase e "viste" per la CNN.

Pipeline (stile AstroNet, ricostruita da zero e semplificata):
1. normalizza il flusso e rimuove il trend lento con una mediana mobile
2. cerca il segnale periodico con Box Least Squares (se non hai gia' un'effemeride)
3. ripete il detrending MASCHERANDO i transiti, altrimenti il filtro li "mangia"
4. ripiega la curva sul periodo e costruisce due viste a lunghezza fissa:
   - globale: l'intera orbita (serve a vedere l'eclisse secondaria a fase 0.5)
   - locale: zoom attorno al transito (serve a vederne la forma)
5. calcola poche feature scalari (profondita', periodo, durata/periodo, SNR) perche'
   normalizzare le viste a minimo -1 cancella l'informazione sulla profondita'
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

GLOBAL_BINS = 2001
LOCAL_BINS = 201
N_EXTRA = 4
DEFAULT_DURATIONS = (0.04, 0.08, 0.12, 0.2, 0.3)  # giorni


def wrapped_phase(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Fase in giorni, nell'intervallo [-period/2, period/2), centrata sul transito."""
    return ((time - t0 + 0.5 * period) % period) - 0.5 * period


def robust_sigma(x: np.ndarray) -> float:
    """Deviazione standard robusta (MAD scalata)."""
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def normalize_flux(flux: np.ndarray) -> np.ndarray:
    return flux / np.nanmedian(flux)


def running_median(
    time: np.ndarray, flux: np.ndarray, window_days: float, use: np.ndarray | None = None
) -> np.ndarray:
    """Mediana mobile calcolata in tempo (robusta ai buchi) e valutata su tutti i punti.

    `use` e' una maschera booleana dei punti ammessi alla stima del trend; il trend nei
    punti esclusi (es. dentro i transiti) viene interpolato linearmente.
    """
    if use is None:
        use = np.ones(time.size, dtype=bool)
    t, f = time[use], flux[use]
    if t.size < 10:
        raise ValueError("Troppi pochi punti per stimare il trend")
    lo = np.searchsorted(t, t - window_days / 2.0, side="left")
    hi = np.searchsorted(t, t + window_days / 2.0, side="right")
    trend = np.array([np.median(f[a:b]) for a, b in zip(lo, hi, strict=True)])
    return np.interp(time, t, trend)


def transit_mask(
    time: np.ndarray, period: float, t0: float, duration: float, scale: float = 1.5
) -> np.ndarray:
    """True per i punti entro `scale` volte la durata del transito."""
    return np.abs(wrapped_phase(time, period, t0)) < scale * duration / 2.0


def count_observed_transits(
    time: np.ndarray, period: float, t0: float, duration: float, min_points: int = 3
) -> int:
    """Quante epoche previste hanno davvero dati: almeno `min_points` cadenze entro duration/2.

    Un transito che cade in un buco della curva (tra trimestri, safe mode) non conta.
    """
    epoch = np.round((time - t0) / period)
    inside = np.abs(time - (t0 + epoch * period)) < duration / 2.0
    _, counts = np.unique(epoch[inside], return_counts=True)
    return int((counts >= min_points).sum())


def detrend(
    time: np.ndarray, flux: np.ndarray, window_days: float = 1.5, mask: np.ndarray | None = None
) -> np.ndarray:
    """Restituisce flusso/trend (circa 1). `mask` esclude dal trend i punti in transito."""
    flux = normalize_flux(flux)
    use = None if mask is None else ~mask
    return flux / running_median(time, flux, window_days, use)


def clip_upward(flux: np.ndarray, nsigma: float = 5.0) -> np.ndarray:
    """Maschera dei punti da tenere: scarta solo gli outlier verso l'alto (i dip sono segnale)."""
    sigma = robust_sigma(flux)
    return flux < np.median(flux) + nsigma * sigma


@dataclass
class Ephemeris:
    period: float  # giorni
    t0: float  # giorni (stessa scala di `time`)
    duration: float  # giorni
    depth: float  # frazione del flusso
    snr: float


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float | None = None,
    durations: tuple[float, ...] = DEFAULT_DURATIONS,
    max_periods: int = 20000,
) -> Ephemeris:
    """Box Least Squares su curva gia' appiattita. Griglia di periodi uniforme in log."""
    from astropy.timeseries import BoxLeastSquares

    baseline = float(time.max() - time.min())
    if max_period is None:
        max_period = baseline / 3.0  # almeno 3 transiti
    if max_period <= min_period:
        raise ValueError("Baseline troppo corta per l'intervallo di periodi richiesto")
    durs = np.array([d for d in durations if d < min_period], dtype=float)
    if durs.size == 0:
        raise ValueError("Nessuna durata inferiore a min_period")

    resolution = 0.5 * durs.min() / baseline
    n = int(np.ceil(np.log(max_period / min_period) / resolution))
    n = min(max(n, 1000), max_periods)
    periods = np.exp(np.linspace(np.log(min_period), np.log(max_period), n))

    dy = np.full(time.size, robust_sigma(flux))
    result = BoxLeastSquares(time, flux, dy=dy).power(periods, durs, objective="snr")
    i = int(np.argmax(result.power))
    return Ephemeris(
        period=float(result.period[i]),
        t0=float(result.transit_time[i]),
        duration=float(result.duration[i]),
        depth=float(result.depth[i]),
        snr=float(result.depth_snr[i]),
    )


def bin_view(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float, num_bins: int):
    """Mediana di `y` in `num_bins` intervalli uguali su [x_min, x_max].

    I bin vuoti sono riempiti per interpolazione lineare; se tutti vuoti restituisce zeri.
    """
    edges = np.linspace(x_min, x_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    inside = (x >= x_min) & (x <= x_max)
    idx = np.clip(np.digitize(x[inside], edges) - 1, 0, num_bins - 1)
    yi = y[inside]
    order = np.argsort(idx, kind="stable")
    idx, yi = idx[order], yi[order]
    bins = np.arange(num_bins)
    starts = np.searchsorted(idx, bins, side="left")
    ends = np.searchsorted(idx, bins, side="right")

    out = np.full(num_bins, np.nan)
    for b in range(num_bins):
        if ends[b] > starts[b]:
            out[b] = np.median(yi[starts[b] : ends[b]])
    good = ~np.isnan(out)
    if not good.any():
        return np.zeros(num_bins)
    return np.interp(centers, centers[good], out[good])


def normalize_view(view: np.ndarray) -> np.ndarray:
    """Mediana a 0 e minimo a -1 (se la vista non e' piatta)."""
    v = view - np.median(view)
    depth = -v.min()
    return v / depth if depth > 1e-12 else v


@dataclass
class CandidateViews:
    global_view: np.ndarray  # (GLOBAL_BINS,)
    local_view: np.ndarray  # (LOCAL_BINS,)
    extra: np.ndarray  # (N_EXTRA,)


def make_views(
    time: np.ndarray,
    flux: np.ndarray,
    eph: Ephemeris,
    global_bins: int = GLOBAL_BINS,
    local_bins: int = LOCAL_BINS,
    local_half_width_durations: float = 2.0,
) -> CandidateViews:
    """Costruisce vista globale, vista locale e feature scalari da una curva appiattita."""
    phase = wrapped_phase(time, eph.period, eph.t0)
    g = bin_view(phase, flux, -eph.period / 2, eph.period / 2, global_bins)
    half = local_half_width_durations * eph.duration
    l_view = bin_view(phase, flux, -half, half, local_bins)
    extra = np.array(
        [
            np.log10(max(eph.depth, 1e-6)),
            np.log10(eph.period),
            eph.duration / eph.period,
            np.log1p(max(eph.snr, 0.0)),
        ],
        dtype=np.float32,
    )
    return CandidateViews(
        normalize_view(g).astype(np.float32), normalize_view(l_view).astype(np.float32), extra
    )


def ephemeris_from_extra(extra: np.ndarray) -> Ephemeris:
    """Inverte le feature scalari di `make_views` (t0 non recuperabile: vale 0).

    Serve per generare testi/report dai soli dataset .npz; non per la classificazione.
    """
    period = 10 ** float(extra[1])
    return Ephemeris(
        period=period,
        t0=0.0,
        duration=float(extra[2]) * period,
        depth=10 ** float(extra[0]),
        snr=float(np.expm1(extra[3])),
    )


def process_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    ephemeris: Ephemeris | None = None,
    window_days: float = 1.5,
    exclude: Sequence[Ephemeris] = (),
    **bls_kwargs,
) -> tuple[CandidateViews, Ephemeris]:
    """Dalla curva grezza alle viste. Se `ephemeris` e' None la cerca con BLS.

    Con effemeride nota (es. KOI della NASA) si salta il BLS: come fa AstroNet con i TCE.
    `exclude`: effemeridi di ALTRI segnali della stessa stella; i punti nei loro transiti vengono
    eliminati, cosi' non appaiono come dip spuri nelle viste del candidato.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(time) & np.isfinite(flux)
    order = np.argsort(time[ok])
    time, flux = time[ok][order], flux[ok][order]
    for other in exclude:
        keep_other = ~transit_mask(time, other.period, other.t0, other.duration)
        time, flux = time[keep_other], flux[keep_other]

    eph = ephemeris
    if eph is None:
        flat = detrend(time, flux, window_days)
        keep = clip_upward(flat)
        eph = run_bls(time[keep], flat[keep], **bls_kwargs)

    mask = transit_mask(time, eph.period, eph.t0, eph.duration)
    flat = detrend(time, flux, max(window_days, 3 * eph.duration), mask=mask)
    keep = clip_upward(flat)
    return make_views(time[keep], flat[keep], eph), eph
