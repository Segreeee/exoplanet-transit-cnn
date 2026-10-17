"""Esporta i dati della demo web (web/public/data/demo.json) da risultati su dati Kepler REALI.

    python scripts/export_web_data.py --data data/processed/kepler.npz

Esempi: scelti da `webdata.select_examples` tra i KOI del TEST del seed 0 (il modello `no_scalar` di
quel seed non li ha visti; split per stella). Gli errori sono sovra-rappresentati: la lista non e'
un campione rappresentativo, le metriche vengono dal test set intero (5 seed). Ogni numero della UI
viene da qui o da reports/*.json: nulla e' scritto a mano nel codice web.
Prima di scrivere il file controlla che i punteggi salvati coincidano con quelli del modello.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))  # per importare gli script fratelli
from exotransit.experiments import load_model  # noqa: E402
from exotransit.koi import prepare_koi_table  # noqa: E402
from exotransit.llm import template_report  # noqa: E402
from exotransit.webdata import bin_window, transit_window  # noqa: E402
from generate_vetting_texts import demo_selection  # noqa: E402


def nan_to_none(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick(d: dict, *keys):
    return {k: d[k] for k in keys}


def results_block(args) -> dict:
    """Numeri dei risultati, copiati dai report (nessun ricalcolo qui)."""
    cnn, base = load(args.cnn_report), load(args.baselines_report)
    err, inj, rob = load(args.errors_report), load(args.injection_report), load(args.robust_report)
    out = {
        "kepler_dataset": pick(
            load(args.kepler_dataset),
            "n_examples",
            "n_stars",
            "examples_per_class",
            "stars_per_class",
            "n_stars_with_multiple_kois",
        ),
        "baselines": {
            "splits": base["splits"],
            "models": {
                k: {
                    "test": {
                        m: v["test"][m] for m in ("roc_auc", "pr_auc", "precision_at_recall90", "n")
                    }
                    | {"ci95": v["test"]["ci95"]}
                }
                for k, v in base["models"].items()
            },
        },
        "cnn": {
            "n_seeds": len(cnn["args"]["seeds"]),
            "summary": {
                k: {"metrics": v["metrics"], "paired_diff_vs_rf": v["paired_diff_vs_rf"]}
                for k, v in cnn["summary"].items()
            },
            "baseline_random_forest": cnn["baseline_random_forest_scalar"],
            "test_sizes": [r["test"]["n"] for r in cnn["runs"] if r["variant"] == "no_scalar"],
        },
        "regimes": {
            "n_rows": err["n_rows"],
            "n_unique_kois": err["n_unique_kois"],
            "tables": err["tables"],
            "calibration": err["calibration"]["summary"],
        },
        "injection": pick(inj, "depths_ppm", "periods", "n_injections", "n_valid")
        | {"cells": [pick(c, "depth_ppm", "period", "mean", "std", "n") for c in inj["cells"]]},
        "robustness": rob["summary"],
    }
    if Path(args.tess_transfer).exists():
        tt, td = load(args.tess_transfer), load(args.tess_dataset)
        whole = [v["test"] for v in tt["kepler_as_is_all_tess"].values()]
        out["tess"] = {
            "dataset": pick(td, "n_examples", "n_stars", "examples_per_class", "discarded"),
            "n_seeds": len(tt["runs"]),
            "test_sizes": [r["split"]["n_test"] for r in tt["runs"]],
            "summary": {
                k: {"metrics": v["metrics"], "paired_diff": v["paired_diff"]}
                for k, v in tt["summary"].items()
            },
            "kepler_as_is_all_tess": {
                "n": whole[0]["n"],
                "prevalence": whole[0]["prevalence"],
                "roc_auc_range": [
                    min(w["roc_auc"] for w in whole),
                    max(w["roc_auc"] for w in whole),
                ],
                "pr_auc_range": [min(w["pr_auc"] for w in whole), max(w["pr_auc"] for w in whole)],
            },
        }
    if Path(args.vetting_v2).exists():
        v1, v2 = load(args.vetting_v1), load(args.vetting_v2)
        out["vetting"] = {
            "n_candidates": v2["n_candidates"],
            "prompt_v1": {m: r["summary_rechecked"] for m, r in v1["models"].items()},
            "prompt_v2": {m: r["summary"] for m, r in v2["models"].items()},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cnn-report", default="reports/kepler_cnn.json")
    ap.add_argument("--variant", default="no_scalar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models-dir", default="models/kepler")
    ap.add_argument("--koi", default="data/raw/koi.csv")
    ap.add_argument("--cache", default="data/raw/kepler")
    ap.add_argument("--texts", default="reports/web_vetting_texts.json")
    ap.add_argument("--kepler-dataset", default="reports/kepler_dataset.json")
    ap.add_argument("--baselines-report", default="reports/kepler_baselines.json")
    ap.add_argument("--errors-report", default="reports/kepler_errors.json")
    ap.add_argument("--injection-report", default="reports/kepler_injection.json")
    ap.add_argument("--robust-report", default="reports/kepler_robustness.json")
    ap.add_argument("--tess-transfer", default="reports/tess_transfer.json")
    ap.add_argument("--tess-dataset", default="reports/tess_dataset.json")
    ap.add_argument("--vetting-v1", default="reports/vetting_llm_comparison.json")
    ap.add_argument("--vetting-v2", default="reports/vetting_llm_comparison_v2.json")
    ap.add_argument("--out", default="web/public/data/demo.json")
    args = ap.parse_args()

    with np.load(args.data) as d:
        g, loc, extra, y, groups, names = (
            d[k] for k in ("global_view", "local_view", "extra", "y", "groups", "names")
        )
    data = {"y": y, "extra": extra, "names": names}
    run = next(
        r
        for r in load(args.cnn_report)["runs"]
        if r["variant"] == args.variant and r["seed"] == args.seed
    )
    picks, facts_all, scores = demo_selection(data, run)
    thr = run["threshold_from_val"]
    idx = np.array([p["index"] for p in picks])

    # coerenza: i punteggi salvati devono coincidere con quelli ricalcolati dal modello salvato
    model, model_thr = load_model(Path(args.models_dir) / f"{args.variant}_seed{args.seed}.pt")
    again = model.predict_proba(g[idx], loc[idx], extra[idx])
    if not np.allclose(again, scores[idx], atol=1e-3) or abs(model_thr - thr) > 1e-9:
        raise SystemExit("Punteggi o soglia salvati non coincidono col modello: export interrotto.")

    kois = prepare_koi_table(pd.read_csv(args.koi)).set_index("name")
    texts = load(args.texts) if Path(args.texts).exists() else {"model": None, "texts": {}}
    examples = []
    for p, facts in zip(picks, facts_all, strict=True):
        i, name = p["index"], str(names[p["index"]])
        row = kois.loc[name]
        with np.load(Path(args.cache) / f"{int(groups[i])}.npz") as c:
            time, flux = c["time"], c["flux"]
        start, end, transits = transit_window(time, row["period"], row["t0"], row["duration"])
        card = texts["texts"].get(name)
        examples.append(
            {
                "id": name,
                "kepid": int(groups[i]),
                "label": int(y[i]),
                "outcome": p["outcome"],
                "score": round(float(scores[i]), 4),
                "prob_calibrated": facts["probabilita_pianeta"],
                "predicted_candidate": bool(scores[i] >= thr),
                "period_days": facts["periodo_giorni"],
                "duration_hours": facts["durata_ore"],
                "depth_ppm": facts["profondita_ppm"],
                "snr": facts["snr"],
                "report": card["text"] if card else template_report(facts),
                "report_source": card["source"] if card else "template",
                "light_curve": bin_window(time, flux / np.nanmedian(flux), start, end, 1200),
                "transits": transits,
                "global_view": np.round(g[i], 3).tolist(),
                "local_view": np.round(loc[i], 3).tolist(),
            }
        )

    ds = load(args.kepler_dataset)
    out = {
        "meta": {
            "data_kind": "kepler_koi",
            "model": f"CNN {args.variant}, seed {args.seed}",
            "n_seeds": len(load(args.cnn_report)["args"]["seeds"]),
            "threshold_score": round(float(thr), 4),
            "n_examples_shown": len(examples),
            "shown_by_outcome": {
                k: sum(e["outcome"] == k for e in examples) for k in ("TP", "TN", "FP", "FN")
            },
            "test_set_size": len(run["test_idx"]),
            "dataset_examples": ds["n_examples"],
            "report_model": texts["model"],
            "n_template_reports": sum(e["report_source"] == "template" for e in examples),
            "torch": torch.__version__,
        },
        "results": results_block(args),
        "examples": examples,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nan_to_none(out), separators=(",", ":")), encoding="utf-8")
    print(f"Scritto {path} ({path.stat().st_size / 1024:.0f} KB, {len(examples)} esempi del test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
