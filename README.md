# exoplanet-transit-cnn

Classificazione di candidati esopianeta (pianeta vs falso positivo) da curve di luce Kepler con una
CNN a due viste (globale + locale) e feature scalari, piu' schede di vetting in italiano generate da
un LLM locale (Ollama) a partire da numeri gia' calcolati. Progetto personale da portfolio.

## Stato (leggi prima di fidarti)
- **Verificato**: pipeline completa su **dati sintetici** (generazione, BLS, viste, training con split
  per stella, baseline random forest, metriche) e test automatici.
- **Verificato sui dati reali (solo costruzione del dataset)**: download Kepler di tutte le 6424 stelle con KOI
  CONFIRMED/FALSE POSITIVE e `data/processed/kepler.npz` (7181 esempi, 2709 positivi e 4472 falsi positivi; vedi
  `reports/kepler_dataset.json`). Ispezione visiva di 20 curve senza anomalie sistematiche.
  La GPU e' stata usata nello smoke test sintetico (`reports/synthetic_metrics.json`).
  **Non verificato**: uso di Ollama con un modello reale. **Nessun modello e' stato ancora addestrato su dati
  reali**: prossimo passo, Fase 2 (baseline) di `docs/ROADMAP.md`.
- **Soglia di decisione**: a soglia 0.5 la CNN sintetica non classificava nessun pianeta (punteggi non calibrati);
  la soglia ora si sceglie sulla validazione. Alla soglia scelta, la baseline random forest classifica in modo
  simile alla CNN: il vantaggio della CNN e' solo sul ranking, non dimostrato.
- **Nessun risultato su dati reali.** I numeri sintetici dimostrano che il codice funziona, non che il modello
  funzioni su Kepler. Il test set sintetico era di 30 esempi: differenze piccole non sono significative.

## Setup
Richiede Python 3.10+ e, per la GPU, un driver NVIDIA recente.
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# 1) PyTorch con CUDA: scegli il comando su https://pytorch.org/get-started/locally/
# 2) poi:
pip install -e ".[dev]"            # aggiungi ,kepler per i dati reali: pip install -e ".[dev,kepler]"
python scripts/check_env.py
pytest
```
PyTorch non e' tra le dipendenze del pacchetto apposta: la build giusta dipende dalla tua GPU/driver.
Con una RTX 3050 e' normale partire dal batch size 64; se finisci la VRAM scendi con `--batch-size 32`.

## Quickstart (sintetico)
```
python scripts/build_dataset.py --source synthetic --n 300 --workers 4 --out data/processed/synthetic.npz
python scripts/train.py --data data/processed/synthetic.npz --epochs 25 \
    --out models/synthetic_cnn.pt --report reports/synthetic_metrics.json
python scripts/vetting_report.py --data data/processed/synthetic.npz --model models/synthetic_cnn.pt --no-llm
```
Con Ollama: `export OLLAMA_MODEL=<modello da ollama list>` e togli `--no-llm`.

## Demo web e controlli
```
python scripts/export_web_data.py --data data/processed/synthetic.npz --model models/synthetic_cnn.pt \
    --report reports/synthetic_metrics.json --n-total 300
cd web && npm install && npm run dev
python scripts/sanity_checks.py --data data/processed/synthetic.npz --model models/synthetic_cnn.pt \
    --report reports/synthetic_metrics.json --demo web/public/data/demo.json
```
Deploy su Vercel: `docs/DEPLOY_VERCEL.md`. Come verificare il lavoro degli agenti: `docs/VERIFYING_AGENT_WORK.md`.

## Dati reali
```
pip install -e ".[kepler]"
python scripts/download_kepler.py --limit 50
python scripts/build_dataset.py --source kepler --koi data/raw/koi.csv --cache data/raw/kepler     --out data/processed/kepler.npz --report reports/kepler_dataset.json
python scripts/plot_kepler_examples.py --kepids 3642289   # figure di ispezione in reports/figures/
```
Un esempio per KOI (gruppo = `kepid`, lo split resta per stella); i transiti degli altri KOI della stessa
stella vengono mascherati; scartati i KOI senza curva, senza SNR o con meno di 3 transiti osservati
(`--min-transits`). Nessun taglio sull'SNR: si filtra dopo da `extra[:, 3]`. Il report JSON elenca i conteggi
e i motivi di scarto.

## Struttura
Vedi `CLAUDE.md` (regole e convenzioni), `docs/ROADMAP.md` (fasi), `docs/PROMPTS.md` (prompt per Claude
Code), `docs/SCIENCE_NOTES.md`, `docs/PLUGINS_AND_OLLAMA.md`.

## Come lavorare con Claude Code
```
cd exoplanet-transit-cnn
git init && git add . && git commit -m "Starter kit"
claude
```
Poi incolla il prompt della Fase 0 da `docs/PROMPTS.md`.

## Limiti
Classificatore sulla sola curva di luce: niente centroide, spettri, immagini. Le etichette KOI sono
imperfette. L'architettura e' ispirata ad AstroNet (Shallue & Vanderburg 2018) ma e' una
re-implementazione piu' piccola, non una riproduzione dei suoi risultati.
