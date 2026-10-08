"""Esporta i dati per la demo web (web/public/data/demo.json).

    python scripts/export_web_data.py --data data/processed/synthetic.npz \
        --model models/synthetic_cnn.pt --report reports/synthetic_metrics.json --n-total 300

Usa SOLO esempi del test set (stesso split per stella del training). Le curve grezze non sono nel
.npz: si rigenerano con make_dataset(n_total, seed) e si verifica che coincidano con il dataset
(stessa stella e stessa classe). Se non coincidono lo script si ferma: meglio nessun export che
un export incoerente con le metriche.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exotransit.llm import candidate_facts, template_report
from exotransit.model import TwoTowerCNN
from exotransit.preprocess import ephemeris_from_extra
from exotransit.splits import split_by_group
from exotransit.synthetic import make_dataset


def downsample(time: np.ndarray, flux: np.ndarray, n: int) -> dict:
    """Media in n intervalli uguali (basta per mostrare la curva nel browser)."""
    edges = np.linspace(time.min(), time.max(), n + 1)
    idx = np.clip(np.digitize(time, edges) - 1, 0, n - 1)
    counts = np.bincount(idx, minlength=n)
    sums = np.bincount(idx, weights=flux, minlength=n)
    keep = counts > 0
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "t": np.round(centers[keep], 3).tolist(),
        "f": np.round(sums[keep] / counts[keep], 5).tolist(),
    }


def nan_to_none(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-total", type=int, required=True, help="n usato in build_dataset")
    ap.add_argument("--data-seed", type=int, default=0, help="--seed usato in build_dataset")
    ap.add_argument("--out", default="web/public/data/demo.json")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    split_seed = report["args"]["seed"]
    with np.load(args.data, allow_pickle=False) as d:
        g, loc, extra, y, groups, names = (
            d[k] for k in ("global_view", "local_view", "extra", "y", "groups", "names")
        )
    if len(y) != args.n_total:
        raise SystemExit(f"Il dataset ha {len(y)} esempi, ma --n-total e' {args.n_total}.")

    raw = make_dataset(args.n_total, seed=args.data_seed)
    same = all(
        r.star_id == int(gr) and r.kind == str(nm) and r.label == int(lab)
        for r, gr, nm, lab in zip(raw, groups, names, y, strict=True)
    )
    if not same:
        raise SystemExit("Le curve rigenerate non coincidono col dataset: controlla seed e n.")

    test_idx = split_by_group(y, groups, seed=split_seed)["test"]
    ckpt = torch.load(args.model, map_location="cpu")
    model = TwoTowerCNN(base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = float(ckpt["threshold"])  # scelta sulla validazione in train.py
    probs = model.predict_proba(g[test_idx], loc[test_idx], extra[test_idx])

    examples = []
    for i, p in zip(test_idx, probs, strict=True):
        eph = ephemeris_from_extra(extra[i])
        facts = candidate_facts(
            f"{names[i]}-{groups[i]}", eph, float(p), threshold=round(threshold, 4)
        )
        examples.append(
            {
                "id": f"{names[i]}-{groups[i]}",
                "kind": str(names[i]),
                "label": int(y[i]),
                "prob_planet": round(float(p), 4),
                "predicted_candidate": bool(p >= threshold),
                "period_days": facts["periodo_giorni"],
                "duration_hours": facts["durata_ore"],
                "depth_ppm": facts["profondita_ppm"],
                "snr": facts["snr_bls"],
                "report": template_report(facts),
                "light_curve": downsample(raw[i].time, raw[i].flux, 700),
                "global_view": np.round(g[i], 3).tolist(),
                "local_view": np.round(loc[i], 3).tolist(),
            }
        )

    out = {
        "meta": {
            "data_kind": "synthetic",
            "n_examples_total": int(len(y)),
            "split_sizes": report["split_sizes"],
            "seed": split_seed,
            "device": report["device"],
            "epochs": report["args"]["epochs"],
            "best_epoch": report["best_epoch"],
            "torch": torch.__version__,
            "threshold_cnn": round(threshold, 4),
            "threshold_rf": report["threshold_rf_from_val"],
        },
        "metrics": {
            "cnn_default_threshold": report["test_cnn"],
            "cnn_val_threshold": report["test_cnn_at_val_threshold"],
            "rf_default_threshold": report["test_baseline_random_forest_on_scalar_features"],
            "rf_val_threshold": report["test_baseline_rf_at_val_threshold"],
        },
        "examples": examples,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON valido per il browser: NaN -> null
    path.write_text(json.dumps(nan_to_none(out), separators=(",", ":")))
    print(f"Scritto {path} ({path.stat().st_size / 1024:.0f} KB, {len(examples)} esempi di test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
