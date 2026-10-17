"""Controlli indipendenti sulla demo web: i numeri che mostra reggono? Esce con 1 se uno FALLISCE.

    python scripts/check_demo.py --data data/processed/kepler.npz

Non si fida di export_web_data.py: ricalcola i punteggi dal modello salvato sulle viste NON
arrotondate, confronta i blocchi dei risultati con i report originali, verifica che gli esempi
siano del test del seed 0 e che ogni testo dell'LLM passi i controlli di `llm.check_report`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from exotransit.experiments import load_model
from exotransit.preprocess import GLOBAL_BINS, LOCAL_BINS
from exotransit.webdata import calibrated_facts

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_bad_number(obj) -> bool:
    if obj is None:
        return True
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(has_bad_number(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_bad_number(v) for v in obj)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--demo", default="web/public/data/demo.json")
    ap.add_argument("--cnn-report", default="reports/kepler_cnn.json")
    ap.add_argument("--errors-report", default="reports/kepler_errors.json")
    ap.add_argument("--tess-transfer", default="reports/tess_transfer.json")
    ap.add_argument("--models-dir", default="models/kepler")
    args = ap.parse_args()

    demo, cnn = load(args.demo), load(args.cnn_report)
    ex = demo["examples"]
    with np.load(args.data) as d:
        g, loc, extra, y, names = (
            d[k] for k in ("global_view", "local_view", "extra", "y", "names")
        )
    variant, seed = "no_scalar", 0
    run = next(r for r in cnn["runs"] if r["variant"] == variant and r["seed"] == seed)
    thr = demo["meta"]["threshold_score"]

    # 1. forma dei dati
    record(
        "nessun NaN/null negli esempi", not any(has_bad_number(e) for e in ex), f"{len(ex)} esempi"
    )
    record(
        "lunghezze delle viste",
        all(
            len(e["global_view"]) == GLOBAL_BINS and len(e["local_view"]) == LOCAL_BINS for e in ex
        ),
        f"{GLOBAL_BINS}/{LOCAL_BINS} bin",
    )
    record("id unici", len({e["id"] for e in ex}) == len(ex), f"{len({e['id'] for e in ex})} id")

    # 2. gli esempi sono del TEST del seed 0 (mai visti dal modello)
    test_names = {str(names[i]) for i in run["test_idx"]}
    train_val = {str(names[i]) for i in run["val_idx"]}
    record(
        "esempi solo dal test set",
        all(e["id"] in test_names for e in ex) and not any(e["id"] in train_val for e in ex),
        f"{sum(e['id'] in test_names for e in ex)}/{len(ex)} nel test",
    )

    # 3. coerenza interna: soglia, esito, etichetta
    pred_ok = all(e["predicted_candidate"] == (e["score"] >= thr) for e in ex)
    out_ok = all(
        e["outcome"]
        == (
            "TP"
            if e["label"] and e["predicted_candidate"]
            else "FN"
            if e["label"]
            else "FP"
            if e["predicted_candidate"]
            else "TN"
        )
        for e in ex
    )
    record("esito = etichetta + soglia", pred_ok and out_ok, f"soglia {thr:.4f}")
    lab_ok = all(int(y[np.where(names == e["id"])[0][0]]) == e["label"] for e in ex)
    record("etichette = dataset", lab_ok, "")

    # 4. punteggi ricalcolati dal modello su viste non arrotondate
    model, model_thr = load_model(Path(args.models_dir) / f"{variant}_seed{seed}.pt")
    idx = np.array([np.where(names == e["id"])[0][0] for e in ex])
    again = model.predict_proba(g[idx], loc[idx], extra[idx])
    diff = float(np.max(np.abs(again - np.array([e["score"] for e in ex]))))
    record(
        "punteggi = ricalcolo dal modello",
        diff < 1e-3 and abs(model_thr - thr) < 1e-3,
        f"scarto massimo {diff:.5f}, soglia modello {model_thr:.4f}",
    )

    # 5. i blocchi dei risultati coincidono con i report
    res = demo["results"]
    ref = cnn["summary"][variant]["metrics"]["pr_auc"]["mean"]
    got = res["cnn"]["summary"][variant]["metrics"]["pr_auc"]["mean"]
    record("PR-AUC CNN = report", abs(ref - got) < 1e-12, f"{got:.4f}")
    err = load(args.errors_report)
    record(
        "tabelle per fascia = report",
        res["regimes"]["tables"] == err["tables"] and res["regimes"]["n_rows"] == err["n_rows"],
        f"{len(err['tables'])} tabelle",
    )
    if Path(args.tess_transfer).exists() and "tess" in res:
        tt = load(args.tess_transfer)
        a = tt["summary"]["fine_tuned"]["metrics"]["pr_auc"]["mean"]
        b = res["tess"]["summary"]["fine_tuned"]["metrics"]["pr_auc"]["mean"]
        record("PR-AUC TESS = report", abs(a - b) < 1e-12, f"{b:.4f}")
    record("nessun NaN/null nei risultati", not has_bad_number({k: v for k, v in res.items()}), "")

    # 6. testi: dell'LLM solo se passano i controlli, altrimenti template
    from exotransit.llm import check_report

    facts = calibrated_facts(
        names[idx], extra[idx], again, run["val_scores"], y[run["val_idx"]], thr
    )
    bad = []
    for e, f in zip(ex, facts, strict=True):
        if e["report_source"] != "template" and check_report(e["report"], f):
            bad.append(e["id"])
    n_llm = sum(e["report_source"] != "template" for e in ex)
    record("testi dell'LLM senza difetti noti", not bad, f"{n_llm} testi LLM, problemi: {bad[:5]}")
    sources = {e["report_source"] for e in ex} - {"template"}
    record(
        "modello dei testi dichiarato",
        sources <= {demo["meta"]["report_model"]},
        f"modelli nei testi: {sorted(sources)}",
    )

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(failed)} controlli falliti su {len(RESULTS)}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
