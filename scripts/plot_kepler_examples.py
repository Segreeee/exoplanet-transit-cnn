"""Conteggi per classe delle curve scaricate e figure di ispezione (curva grezza -> viste).

    python scripts/plot_kepler_examples.py --kepids 3642289 4940438 8098728

Per ogni stella usa il KOI con SNR piu' alto e salva reports/figures/kepid_{kepid}.png con:
grezza + trend, appiattita (transiti evidenziati), ripiegata, vista globale, vista locale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exotransit.koi import prepare_koi_table, row_to_ephemeris, sibling_ephemerides
from exotransit.preprocess import (
    clip_upward,
    detrend,
    make_views,
    normalize_flux,
    running_median,
    transit_mask,
    wrapped_phase,
)


def pick_examples(sub: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Fino a n/2 stelle per classe (a caso, con seed), col KOI a SNR piu' alto di ciascuna."""
    best = sub.sort_values("snr", ascending=False).drop_duplicates("kepid")
    parts = [g.sample(n=min(len(g), n // 2), random_state=seed) for _, g in best.groupby("label")]
    return pd.concat(parts).sort_values(["label", "kepid"]).reset_index(drop=True)


def plot_star(row: pd.Series, time: np.ndarray, flux: np.ndarray, siblings, out: Path) -> None:
    """Come `views_for_koi`: dai dati si tolgono prima i transiti degli altri KOI della stella."""
    eph = row_to_ephemeris(row)
    for other in siblings:
        keep_other = ~transit_mask(time, other.period, other.t0, other.duration)
        time, flux = time[keep_other], flux[keep_other]
    mask = transit_mask(time, eph.period, eph.t0, eph.duration)
    window = max(1.5, 3 * eph.duration)
    raw = normalize_flux(flux)
    trend = running_median(time, raw, window, ~mask)
    flat = detrend(time, flux, window, mask=mask)
    keep = clip_upward(flat)
    views = make_views(time[keep], flat[keep], eph)
    phase = wrapped_phase(time[keep], eph.period, eph.t0)

    fig, ax = plt.subplots(5, 1, figsize=(10, 14))
    ax[0].plot(time, raw, ".", ms=1, color="0.5")
    ax[0].plot(
        time, trend, "r-", lw=1, label=f"trend (mediana mobile {window:.1f} d, transiti mascherati)"
    )
    ax[0].set(
        title=f"KIC {row['kepid']} {row['name']} label={row['label']} P={eph.period:.3f} d "
        f"SNR={eph.snr:.1f}",
        ylabel="flusso grezzo",
    )
    ax[0].legend(loc="lower left")
    ax[1].plot(time, flat, ".", ms=1, color="0.5")
    ax[1].plot(
        time[mask], flat[mask], ".", ms=2, color="C1", label="punti mascherati (transito x1.5)"
    )
    ax[1].set(ylabel="appiattita", xlabel="tempo (BKJD, giorni)")
    ax[1].legend(loc="lower left")
    ax[2].plot(phase, flat[keep], ".", ms=1, color="0.6")
    ax[2].set(ylabel="ripiegata", xlabel="fase (giorni)")
    ax[3].plot(views.global_view)
    ax[3].set(ylabel="vista globale", xlabel="bin (2001)")
    ax[4].plot(views.local_view)
    ax[4].set(ylabel="vista locale", xlabel="bin (201)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--koi", default="data/raw/koi.csv")
    ap.add_argument("--cache", default="data/raw/kepler")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--kepids", type=int, nargs="*", default=[])
    ap.add_argument("--n", type=int, default=0, help="campione stratificato di n stelle scaricate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_koi = pd.read_csv(args.koi)
    table = prepare_koi_table(raw_koi)
    all_kois = prepare_koi_table(raw_koi, include_candidates=True)
    have = {int(p.stem) for p in Path(args.cache).glob("*.npz")}
    sub = table[table["kepid"].isin(have)]
    print(
        f"Stelle scaricate: {len(have)}; con KOI utilizzabili (CONFIRMED/FALSE POSITIVE): "
        f"{sub['kepid'].nunique()}"
    )
    names = {1: "CONFIRMED", 0: "FALSE POSITIVE"}
    print("KOI per classe:", {names[k]: int(v) for k, v in sub["label"].value_counts().items()})
    print(
        "Stelle per classe:",
        {names[k]: int(v) for k, v in sub.groupby("label")["kepid"].nunique().items()},
    )

    kepids = list(args.kepids)
    if args.n:
        kepids += pick_examples(sub, args.n, args.seed)["kepid"].tolist()
    for kepid in kepids:
        rows = sub[sub["kepid"] == kepid]
        if rows.empty:
            print(f"KIC {kepid}: nessun KOI utilizzabile o curva non scaricata")
            continue
        row = rows.sort_values("snr", ascending=False).iloc[0]
        with np.load(Path(args.cache) / f"{kepid}.npz") as d:
            time, flux = d["time"], d["flux"]
        out = Path(args.out) / f"kepid_{kepid}.png"
        siblings = sibling_ephemerides(all_kois, int(row["kepid"]), row["name"])
        plot_star(row, time, flux, siblings, out)
        print("salvato", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
