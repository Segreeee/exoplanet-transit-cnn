"""Fase 4.1-4.2: errori per fascia, galleria FN/FP, calibrazione e soglia per recall obiettivo.

    python scripts/analyze_errors.py --data data/processed/kepler.npz \
        --cnn-report reports/kepler_cnn.json --report reports/kepler_errors.json

Usa i punteggi salvati da `run_cnn_experiments.py` (modello `no_scalar`, soglia di ogni seed dalla
validazione). Le righe sono i test dei 5 seed messi insieme: la stessa stella puo' comparire in piu'
seed (split diversi), quindi i conteggi per fascia non sono indipendenti.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from exotransit.errors import binned_rates
from exotransit.koi import prepare_koi_table, sibling_ephemerides
from exotransit.metrics import expected_calibration_error, reliability_curve, threshold_for_recall
from plot_kepler_examples import plot_star

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PERIOD_EDGES = [0, 1.5, 3, 10, 30, 100, 5000]
DEPTH_PPM_EDGES = [0, 200, 500, 1000, 2500, 10000, 1e7]
SNR_EDGES = [0, 7.1, 10, 15, 25, 50, 100, 500, 1e9]
RECALL_TARGETS = (0.90, 0.95)


def pooled_rows(runs, groups, names, y, extra) -> pd.DataFrame:
    rows = []
    for r in runs:
        idx = np.asarray(r["test_idx"])
        rows.append(
            pd.DataFrame(
                {
                    "seed": r["seed"],
                    "kepid": groups[idx],
                    "name": names[idx],
                    "label": y[idx],
                    "score": r["test_scores"],
                    "thr": r["threshold_from_val"],
                    "period": 10 ** extra[idx, 1].astype(float),
                    "depth_ppm": 1e6 * 10 ** extra[idx, 0].astype(float),
                    "snr": np.expm1(extra[idx, 3].astype(float)),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["pred"] = df["score"] >= df["thr"]
    return df


def _mean_std(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}


def calibration(runs, y) -> dict:
    per_seed, pooled = [], {"raw": ([], []), "iso": ([], [])}
    for r in runs:
        yv, sv = y[r["val_idx"]], np.asarray(r["val_scores"])
        yt, st = y[r["test_idx"]], np.asarray(r["test_scores"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(sv, yv)
        ct = iso.predict(st)
        entry = {
            "seed": r["seed"],
            "ece_raw": expected_calibration_error(yt, st),
            "ece_isotonic": expected_calibration_error(yt, ct),
            "brier_raw": float(brier_score_loss(yt, st)),
            "brier_isotonic": float(brier_score_loss(yt, ct)),
            "recall_targets": {},
        }
        for target in RECALL_TARGETS:
            thr = threshold_for_recall(yv, sv, target)
            pred = st >= thr
            tp = int((pred & (yt == 1)).sum())
            entry["recall_targets"][str(target)] = {
                "threshold_from_val": thr,
                "test_recall": tp / max(int((yt == 1).sum()), 1),
                "test_precision": tp / max(int(pred.sum()), 1),
            }
        per_seed.append(entry)
        pooled["raw"][0].append(yt)
        pooled["raw"][1].append(st)
        pooled["iso"][0].append(yt)
        pooled["iso"][1].append(ct)

    summary = {
        k: _mean_std([e[k] for e in per_seed])
        for k in per_seed[0]
        if k.startswith(("ece", "brier"))
    }
    for target in RECALL_TARGETS:
        for k in ("test_recall", "test_precision"):
            vals = [e["recall_targets"][str(target)][k] for e in per_seed]
            summary[f"target{target}_{k}"] = _mean_std(vals)
    return {"per_seed": per_seed, "summary": summary, "pooled": pooled}


def plot_reliability(pooled, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibrazione perfetta")
    for key, label in (("raw", "punteggio grezzo"), ("iso", "isotonica (da validazione)")):
        mp, fr, _ = reliability_curve(
            np.concatenate(pooled[key][0]), np.concatenate(pooled[key][1]), 10
        )
        ax.plot(mp, fr, "o-", label=label)
    ax.set(xlabel="probabilita' prevista", ylabel="frazione di positivi")
    ax.set_title("Reliability (test, 5 seed)")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cnn-report", default="reports/kepler_cnn.json")
    ap.add_argument("--report", default="reports/kepler_errors.json")
    ap.add_argument("--variant", default="no_scalar")
    ap.add_argument("--koi", default="data/raw/koi.csv")
    ap.add_argument("--cache", default="data/raw/kepler")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--n-gallery", type=int, default=12)
    args = ap.parse_args()

    with np.load(args.data) as d:
        groups, names, y, extra = d["groups"], d["names"], d["y"], d["extra"]
    runs = [r for r in json.load(open(args.cnn_report))["runs"] if r["variant"] == args.variant]
    df = pooled_rows(runs, groups, names, y, extra)

    tables = {
        "periodo_giorni": binned_rates(df, "period", PERIOD_EDGES),
        "profondita_ppm": binned_rates(df, "depth_ppm", DEPTH_PPM_EDGES),
        "snr_catalogo": binned_rates(df, "snr", SNR_EDGES),
    }
    for name, t in tables.items():
        print(f"\n== per {name} (righe pooled sui seed) ==")
        print(t.round(3).to_string(index=False))

    # galleria: KOI unici, media dei punteggi tra i seed in cui compaiono nel test
    per_koi = (
        df.groupby("name")
        .agg(
            label=("label", "first"),
            score=("score", "mean"),
            pred_rate=("pred", "mean"),
            kepid=("kepid", "first"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    fn = per_koi[(per_koi.label == 1) & (per_koi.pred_rate == 0)].nsmallest(args.n_gallery, "score")
    fp = per_koi[(per_koi.label == 0) & (per_koi.pred_rate == 1)].nlargest(args.n_gallery, "score")
    raw_koi = pd.read_csv(args.koi)
    table = prepare_koi_table(raw_koi)
    all_kois = prepare_koi_table(raw_koi, include_candidates=True)
    for kind, sel in (("FN", fn), ("FP", fp)):
        for _, r in sel.iterrows():
            row = table[table["name"] == r["name"]].iloc[0]
            with np.load(Path(args.cache) / f"{int(r['kepid'])}.npz") as c:
                time, flux = c["time"], c["flux"]
            sib = sibling_ephemerides(all_kois, int(row["kepid"]), row["name"])
            out = Path(args.figures) / "errors" / f"{kind}_{r['name']}.png"
            plot_star(row, time, flux, sib, out)
    print(f"\nGalleria: {len(fn)} FN e {len(fp)} FP in {args.figures}/errors/")

    cal = calibration(runs, y)
    plot_reliability(cal.pop("pooled"), Path(args.figures) / "calibration_reliability.png")
    print("\nCalibrazione e soglie (media +- std sui seed):")
    for k, v in cal["summary"].items():
        print(f"  {k}: {v['mean']:.3f} +- {v['std']:.3f}")

    report = {
        "args": vars(args),
        "n_rows": len(df),
        "n_unique_kois": int(df["name"].nunique()),
        "tables": {k: t.to_dict("records") for k, t in tables.items()},
        "false_negatives": fn[["name", "kepid", "score", "n_seeds"]].to_dict("records"),
        "false_positives": fp[["name", "kepid", "score", "n_seeds"]].to_dict("records"),
        "calibration": cal,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Salvato {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
