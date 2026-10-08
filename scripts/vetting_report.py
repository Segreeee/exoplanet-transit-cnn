"""Schede di vetting per i candidati di un dataset .npz, con classificatore + Ollama.

    export OLLAMA_MODEL=llama3.1:8b        # un modello che hai davvero (ollama list)
    python scripts/vetting_report.py --data data/processed/synthetic.npz \
        --model models/synthetic_cnn.pt --n 5

Il classificatore calcola la probabilita'; l'LLM si limita a riformulare numeri gia' calcolati.
I numeri vengono ricostruiti dalle feature scalari (log10 profondita', log10 periodo,
durata/periodo, log1p SNR): servono solo a produrre il testo, non alla classificazione.
NB: gli esempi sono estratti da TUTTO il dataset (anche training): e' una demo del testo,
non una valutazione.
Senza Ollama (o con numeri sospetti nel testo) si usa il testo deterministico.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from exotransit.llm import OllamaClient, candidate_facts, template_report, vetting_report
from exotransit.model import TwoTowerCNN
from exotransit.preprocess import ephemeris_from_extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ollama-model", default=None, help="altrimenti usa OLLAMA_MODEL")
    ap.add_argument("--no-llm", action="store_true", help="solo testo deterministico")
    args = ap.parse_args()

    with np.load(args.data, allow_pickle=False) as d:
        g, l_, e, names = d["global_view"], d["local_view"], d["extra"], d["names"]
    ckpt = torch.load(args.model, map_location="cpu")
    model = TwoTowerCNN(base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = float(ckpt.get("threshold", 0.5))  # scelta sulla validazione in train.py

    idx = np.random.default_rng(args.seed).choice(len(g), size=min(args.n, len(g)), replace=False)
    probs = model.predict_proba(g[idx], l_[idx], e[idx])
    client = OllamaClient(model=args.ollama_model)

    for i, p in zip(idx, probs, strict=True):
        facts = candidate_facts(
            str(names[i]), ephemeris_from_extra(e[i]), float(p), threshold=round(threshold, 4)
        )
        if args.no_llm:
            res = {"text": template_report(facts), "used_llm": False, "suspicious_numbers": []}
        else:
            res = vetting_report(client, facts)
        print(f"--- {facts['candidato']} (LLM: {res['used_llm']})")
        print(res["text"])
        if res["suspicious_numbers"]:
            print(f"  ATTENZIONE numeri non presenti nei dati: {res['suspicious_numbers']}")
        if res.get("error"):
            print(f"  (fallback: {res['error']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
