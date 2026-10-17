"""Fase 6: confronto di modelli Ollama sulle schede di vetting (50 candidati Kepler).

    python scripts/compare_llm_reports.py --data data/processed/kepler.npz \
        --models gemma3:12b qwen2.5-coder:7b --n 50

I candidati sono estratti a caso dal TEST del seed 0 (il classificatore non li ha visti); i
numeri vengono dalla pipeline (periodo, durata, profondita', SNR dal catalogo, probabilita'
CALIBRATA con regressione isotonica adattata sulla validazione). L'LLM riformula soltanto: non
classifica e non calcola. Per ogni scheda: numeri sospetti, lunghezza, linguaggio da "scoperta",
esito riportato, fallback, tempo. Un modello alla volta (la VRAM e' 4 GB); una chiamata di
riscaldamento e' esclusa dai tempi.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from exotransit.llm import (
    OllamaClient,
    candidate_facts,
    check_report,
    summarize_reports,
    vetting_report,
)
from exotransit.preprocess import ephemeris_from_extra


def build_facts(data: dict, run: dict, idx: np.ndarray) -> list[dict]:
    """Dati dei candidati `idx` con probabilita' calibrata (isotonica) sulla validazione."""
    y, extra, names = data["y"], data["extra"], data["names"]
    pos = {i: k for k, i in enumerate(run["test_idx"])}
    scores = np.asarray(run["test_scores"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        np.asarray(run["val_scores"]), y[run["val_idx"]]
    )
    thr = run["threshold_from_val"]
    out = []
    for i in idx:
        raw = float(scores[pos[i]])
        facts = candidate_facts(
            str(names[i]),
            ephemeris_from_extra(extra[i]),
            float(iso.predict([raw])[0]),
            threshold=round(float(iso.predict([thr])[0]), 3),
        )
        # l'esito viene dal punteggio grezzo e dalla sua soglia: la calibrazione serve al testo
        facts["esito_classificatore"] = "candidato" if raw >= thr else "scartato"
        facts["_etichetta_catalogo"] = int(y[i])  # NON va all'LLM: solo per l'analisi a posteriori
        out.append(facts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cnn-report", default="reports/kepler_cnn.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variant", default="no_scalar")
    ap.add_argument("--seed", type=int, default=0, help="seed del modello/split Kepler")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--report", default="reports/vetting_llm_comparison.json")
    args = ap.parse_args()

    with np.load(args.data) as d:
        data = {k: d[k] for k in ("y", "extra", "names")}
    run = next(
        r
        for r in json.load(open(args.cnn_report))["runs"]
        if r["variant"] == args.variant and r["seed"] == args.seed
    )
    rng = np.random.default_rng(args.sample_seed)
    idx = rng.choice(np.asarray(run["test_idx"]), size=args.n, replace=False)
    facts_all = build_facts(data, run, idx)
    print(
        f"{len(idx)} candidati dal test (seed {args.seed}); "
        f"{sum(f['esito_classificatore'] == 'candidato' for f in facts_all)} con esito 'candidato'",
        flush=True,
    )

    results = {}
    for model in args.models:
        client = OllamaClient(model=model, timeout=args.timeout)
        vetting_report(client, {k: v for k, v in facts_all[0].items() if not k.startswith("_")})
        records = []
        for k, facts in enumerate(facts_all, 1):
            to_llm = {key: v for key, v in facts.items() if not key.startswith("_")}
            t = time.perf_counter()
            res = vetting_report(client, to_llm)
            seconds = time.perf_counter() - t
            flags = check_report(res["text"], to_llm) if res["used_llm"] else []
            records.append(
                {
                    "candidato": facts["candidato"],
                    "etichetta_catalogo": facts["_etichetta_catalogo"],
                    "esito": facts["esito_classificatore"],
                    "text": res["text"],
                    "used_llm": res["used_llm"],
                    "error": res["error"],
                    "flags": flags,
                    "seconds": seconds,
                    "words": len(res["text"].split()),
                }
            )
            print(f"  [{model}] {k}/{len(facts_all)} {seconds:.0f}s flags={flags}", flush=True)
        results[model] = {"summary": summarize_reports(records), "reports": records}
        report = {
            "args": vars(args),
            "n_candidates": len(idx),
            "facts": [{k: v for k, v in f.items()} for f in facts_all],
            "models": results,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print("\nRiepilogo (tassi sulle schede prodotte dall'LLM):")
    for model, r in results.items():
        s = r["summary"]
        print(
            f"{model:22s} fallback {s['fallback_rate']:.0%} "
            f"| con almeno un flag {s['flagged_rate']:.0%} "
            f"| flag { ({k: round(v, 2) for k, v in s['flag_rates'].items()}) } "
            f"| tempo mediano {s['seconds_median']:.0f}s p95 {s['seconds_p95']:.0f}s "
            f"| parole {s['words_mean']:.0f}"
        )
    print(f"Salvato {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
