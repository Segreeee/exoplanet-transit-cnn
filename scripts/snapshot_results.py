"""Copia (ridotti) i report che alimentano i documenti in docs/results/, cosi' sono nel git.

    python scripts/snapshot_results.py

`reports/` e' ignorato dal git (artefatti locali, alcuni da MB). I documenti (README, model card,
post) vengono generati da `scripts/render_docs.py` a partire da QUESTI file: chiunque cloni il
repo puo' rigenerarli e controllare ogni numero. Si ricreano rieseguendo pipeline e script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pick(d: dict, *keys):
    return {k: d[k] for k in keys if k in d}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="docs/results")
    args = ap.parse_args()
    r, out = Path(args.reports), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, obj) -> None:
        (out / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {(out / name).stat().st_size / 1024:.1f} KB")

    cnn = load(r / "kepler_cnn.json")
    save(
        "kepler_cnn_summary.json",
        {
            "seeds": cnn["args"]["seeds"],
            "epochs": cnn["args"]["epochs"],
            "lr": cnn["args"]["lr"],
            "batch_size": cnn["args"]["batch_size"],
            "device": cnn["device"],
            "versions": cnn["versions"],
            "n_examples": cnn["n_examples"],
            "summary": cnn["summary"],
            "baseline_random_forest_scalar": cnn["baseline_random_forest_scalar"],
            "test_sizes": [x["test"]["n"] for x in cnn["runs"] if x["variant"] == "no_scalar"],
            "n_params": {x["variant"]: x["n_params"] for x in cnn["runs"]},
        },
    )
    shuf = load(r / "kepler_cnn_shuffled.json")
    save(
        "kepler_shuffled_summary.json",
        pick(shuf, "args", "summary", "baseline_random_forest_scalar"),
    )
    save("kepler_dataset.json", load(r / "kepler_dataset.json"))
    save("kepler_baselines.json", load(r / "kepler_baselines.json"))
    err = load(r / "kepler_errors.json")
    save(
        "kepler_errors.json",
        {
            "n_rows": err["n_rows"],
            "n_unique_kois": err["n_unique_kois"],
            "tables": err["tables"],
            "calibration": err["calibration"]["summary"],
        },
    )
    inj = load(r / "kepler_injection.json")
    save(
        "kepler_injection.json",
        {
            **pick(inj, "depths_ppm", "periods", "n_injections", "n_valid"),
            "cells": [pick(c, "depth_ppm", "period", "mean", "std", "n") for c in inj["cells"]],
        },
    )
    save("kepler_robustness.json", load(r / "kepler_robustness.json")["summary"])
    save("tess_dataset.json", load(r / "tess_dataset.json"))
    tt = load(r / "tess_transfer.json")
    whole = [v["test"] for v in tt["kepler_as_is_all_tess"].values()]
    save(
        "tess_transfer_summary.json",
        {
            "n_seeds": len(tt["runs"]),
            "test_sizes": [x["split"]["n_test"] for x in tt["runs"]],
            "summary": {k: pick(v, "metrics", "paired_diff") for k, v in tt["summary"].items()},
            "kepler_as_is_all_tess": {
                "n": whole[0]["n"],
                "prevalence": whole[0]["prevalence"],
                "roc_auc_range": [
                    min(w["roc_auc"] for w in whole),
                    max(w["roc_auc"] for w in whole),
                ],
                "pr_auc_range": [min(w["pr_auc"] for w in whole), max(w["pr_auc"] for w in whole)],
            },
            "versions": tt["versions"],
        },
    )
    tshuf = load(r / "tess_transfer_shuffled.json")
    save(
        "tess_shuffled_summary.json",
        {"summary": {k: v["metrics"] for k, v in tshuf["summary"].items()}},
    )
    demo = load(Path("web/public/data/demo.json"))["meta"]
    save(
        "demo_meta.json",
        pick(demo, "n_examples_shown", "shown_by_outcome", "test_set_size", "report_model"),
    )
    v1 = load(r / "vetting_llm_comparison.json")
    v2 = load(r / "vetting_llm_comparison_v2.json")
    save(
        "vetting_summary.json",
        {
            "n_candidates": v2["n_candidates"],
            "prompt_v1": {m: x["summary_rechecked"] for m, x in v1["models"].items()},
            "prompt_v2": {m: x["summary"] for m, x in v2["models"].items()},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
