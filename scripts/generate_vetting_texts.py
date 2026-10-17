"""Schede di vetting precalcolate per gli esempi della demo web (un modello Ollama, prompt v2).

    python scripts/generate_vetting_texts.py --data data/processed/kepler.npz \
        --ollama-model gemma3:12b --out reports/web_vetting_texts.json

Gli esempi sono quelli scelti da `exotransit.webdata.select_examples` sul test del seed 0 (gli
stessi che usa export_web_data.py). Ogni testo dell'LLM passa da `llm.check_report`: se
segnalato, o se Ollama non risponde, si usa il template deterministico (`source: "template"`).
Riprendibile: i candidati gia' presenti nel file di uscita (stesso modello) vengono saltati.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from exotransit.llm import OllamaClient, check_report, template_report, vetting_report
from exotransit.webdata import calibrated_facts, select_examples

N_EACH = {"TP": 18, "TN": 18, "FP": 12, "FN": 12}


def demo_selection(data: dict, run: dict, seed: int = 0, n_each: dict | None = None):
    """Indici (nel dataset) degli esempi della demo e fatti calibrati, dal test di `run`."""
    y, extra, names = data["y"], data["extra"], data["names"]
    test_idx = np.asarray(run["test_idx"])
    scores = np.zeros(len(y))
    scores[test_idx] = run["test_scores"]
    thr = run["threshold_from_val"]
    picks = select_examples(y, scores, thr, test_idx, n_each or N_EACH, seed=seed)
    idx = np.array([p["index"] for p in picks])
    facts = calibrated_facts(
        names[idx], extra[idx], scores[idx], run["val_scores"], y[run["val_idx"]], thr
    )
    return picks, facts, scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cnn-report", default="reports/kepler_cnn.json")
    ap.add_argument("--variant", default="no_scalar")
    ap.add_argument("--seed", type=int, default=0, help="seed del modello/split Kepler")
    ap.add_argument("--ollama-model", default="gemma3:12b")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="reports/web_vetting_texts.json")
    args = ap.parse_args()

    with np.load(args.data) as d:
        data = {k: d[k] for k in ("y", "extra", "names")}
    run = next(
        r
        for r in json.load(open(args.cnn_report))["runs"]
        if r["variant"] == args.variant and r["seed"] == args.seed
    )
    _, facts_all, _ = demo_selection(data, run)

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("model") == args.ollama_model:
            done = prev["texts"]
    client = OllamaClient(model=args.ollama_model, timeout=args.timeout)
    texts = dict(done)
    for k, facts in enumerate(facts_all, 1):
        name = facts["candidato"]
        if name in texts:
            continue
        t = time.perf_counter()
        res = vetting_report(client, facts)
        flags = check_report(res["text"], facts) if res["used_llm"] else ["ollama_non_disponibile"]
        use_llm = res["used_llm"] and not flags
        texts[name] = {
            "text": res["text"] if use_llm else template_report(facts),
            "source": args.ollama_model if use_llm else "template",
            "flags": flags,
            "seconds": round(time.perf_counter() - t, 1),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"model": args.ollama_model, "texts": texts}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[{k}/{len(facts_all)}] {name} {texts[name]['source']} flags={flags}", flush=True)
    n_tpl = sum(v["source"] == "template" for v in texts.values())
    print(f"Fatto. {len(texts)} schede, {n_tpl} con template. Salvato {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
