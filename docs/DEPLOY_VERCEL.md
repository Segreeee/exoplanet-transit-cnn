# Demo web su Vercel

La demo e' un sito **statico** (Vite + React + TypeScript) in `web/`. Legge `web/public/data/demo.json`,
generato dalla pipeline da risultati su **dati Kepler reali**. Niente Python a runtime: PyTorch supera i
limiti di dimensione delle funzioni serverless di Vercel e non serve per mostrare risultati precalcolati.

## Rigenerare i dati
```
# 1) testi delle schede (Ollama, ~33 s a scheda, riprendibile; vedi docs/PLUGINS_AND_OLLAMA.md)
python scripts/generate_vetting_texts.py --data data/processed/kepler.npz --ollama-model gemma3:12b
# 2) export: esempi del test del seed 0 + blocco dei risultati copiato da reports/*.json
python scripts/export_web_data.py --data data/processed/kepler.npz
# 3) controllo indipendente (ricalcola i punteggi dal modello, confronta con i report, controlla i testi)
python scripts/check_demo.py --data data/processed/kepler.npz
```
Servono i report `kepler_cnn`, `kepler_baselines`, `kepler_errors`, `kepler_injection`, `kepler_robustness`
(e, se ci sono, `tess_transfer`, `tess_dataset`, `vetting_llm_comparison*`) e i pesi in `models/kepler/`.
L'export si ferma se i punteggi salvati non coincidono con quelli del modello. La UI non contiene numeri scritti a
mano: ogni numero viene dal JSON. Se rigeneri i report, rigenera anche la demo.

## Sviluppo locale
```
cd web && npm install && npm run dev       # http://localhost:5173
npm run build && npm run preview           # prova la build di produzione
```
`demo.json` pesa ~3.5 MB (60 curve di luce con vista globale e locale): committalo, ma non aggiungere altre curve senza
ridurre i punti.

## Deploy
1. Pubblica il repository su GitHub (committa `web/public/data/demo.json`).
2. vercel.com -> Add New -> Project -> importa il repo.
3. **Root Directory: `web`**. Framework Vite (rilevato da `vercel.json`), build `npm run build`, output `dist`.
4. Deploy. Ogni push su `main` ridistribuisce. In alternativa dalla CLI: `cd web && npx vercel`.

### Errore "No python entrypoint found"
Vercel ha guardato la radice del repo, ha trovato `pyproject.toml` e ha creduto a un progetto Python. Il sito e' in
`web/`: imposta **Settings -> General -> Root Directory = `web`** (framework Vite, build `npm run build`, output `dist`)
e rifai il deploy. Se il progetto e' gia' stato creato dalla radice, l'impostazione va cambiata li'. In alternativa un
`vercel.json` nella radice con `"framework": null`, `installCommand`/`buildCommand` con `npm --prefix web ...` e
`outputDirectory` `web/dist` (non provato).

Il deploy **non** e' stato eseguito (nessun account Vercel collegato qui): la build locale e' verificata (`npm run build`),
lo screenshot dell'anteprima e' stato controllato in tema scuro, il tema chiaro e il deploy no. I font (Literata e Schibsted
Grotesk) vengono da Google Fonts; senza rete si usano i font di sistema.

## Demo "live" (inferenza nel browser)
Non fatta, scelta deliberata: la rete e' piccola (~400k parametri) ma il preprocessing (appiattimento con maschera dei
transiti, ripiegamento) andrebbe riscritto in JavaScript o precalcolato, e un'inferenza fragile e' peggio di risultati
precalcolati con gli errori in vista. Se servisse: esportare il modello in ONNX e usare `onnxruntime-web`, oppure un
servizio Python separato chiamato dal sito.
