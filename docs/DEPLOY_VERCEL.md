# Demo web su Vercel

La demo e' un sito **statico** (Vite + React + TypeScript) in `web/`. Legge `web/public/data/demo.json`,
generato dalla pipeline. Niente Python a runtime: PyTorch supera i limiti di dimensione delle funzioni
serverless di Vercel e non serve per mostrare risultati precalcolati.

## Rigenerare i dati
```
python scripts/export_web_data.py --data data/processed/synthetic.npz \
    --model models/synthetic_cnn.pt --report reports/synthetic_metrics.json --n-total 300
python scripts/sanity_checks.py --data data/processed/synthetic.npz \
    --model models/synthetic_cnn.pt --report reports/synthetic_metrics.json \
    --demo web/public/data/demo.json
```
Lo script di export si ferma se il dataset non corrisponde alle curve rigenerate. La UI non contiene
numeri scritti a mano: tutto viene dal JSON. Quando avrai dati Kepler reali, riesporta e cambia
`data_kind` e i testi di avviso (oggi dicono "dati sintetici").

## Sviluppo locale
```
cd web && npm install && npm run dev       # http://localhost:5173
npm run build && npm run preview           # prova la build di produzione
```

## Deploy
1. Pubblica il repository su GitHub (committa `web/public/data/demo.json`: e' piccolo, ~1.6 MB).
2. vercel.com -> Add New -> Project -> importa il repo.
3. **Root Directory: `web`**. Framework Vite (rilevato da `vercel.json`), build `npm run build`, output `dist`.
4. Deploy. Ogni push su `main` ridistribuisce.
In alternativa dalla CLI: `cd web && npx vercel`.

Non ho eseguito il deploy (nessun account Vercel collegato qui): la build locale e' verificata,
il deploy no.

## Se vuoi la demo "live" (inferenza vera nel browser)
Non fatto, non verificato. Opzioni: esportare il modello in ONNX e usare `onnxruntime-web` (la rete e' piccola,
~400k parametri, ma il preprocessing BLS in Python andrebbe riscritto o precalcolato), oppure un servizio
Python separato (es. un Space su Hugging Face) chiamato dal sito. Per un portfolio, i risultati precalcolati
con gli errori in vista sono piu' onesti e piu' robusti di una demo fragile.
