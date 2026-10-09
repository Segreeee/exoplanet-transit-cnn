# exoplanet-transit-cnn: istruzioni per Claude Code

## Obiettivo
Progetto personale da portfolio: classificare segnali di transito di esopianeti (pianeta vs
falso positivo) da curve di luce Kepler, con una CNN a due viste (globale + locale) piu'
feature scalari. Poi, se ha senso, trasferimento a TESS e una piccola demo web.
Chi scrive il progetto viene dall'ingegneria informatica: quando introduci un concetto di
astronomia o fisica, spiegalo in due righe prima di usarlo.

Roadmap: `docs/ROADMAP.md` (aggiorna le checkbox man mano). Prompt pronti: `docs/PROMPTS.md`.
Note scientifiche e riferimenti: `docs/SCIENCE_NOTES.md`.

## Comandi
```
pip install -e ".[dev]"            # installa il pacchetto (PyTorch a parte, vedi README)
python scripts/check_env.py        # librerie, GPU, Ollama
pytest                             # test (devono passare sempre)
ruff check . && ruff format .      # lint e formattazione
python scripts/build_dataset.py --source synthetic --n 300 --workers 4 --out data/processed/synthetic.npz
python scripts/train.py --data data/processed/synthetic.npz --out models/synthetic_cnn.pt --report reports/synthetic_metrics.json
```

## Regole non negoziabili
1. **Split sempre per stella** (`splits.split_by_group`, gruppo = `kepid`). Mai split casuali per
   riga: piu' candidati della stessa stella in split diversi gonfiano le metriche.
2. **Nessun dato o modello nel git** (`data/`, `models/*.pt` sono ignorati). Dati reali mai nei test:
   usa curve sintetiche o fixture minuscole.
3. **Test prima del codice** per ogni logica nuova (`tests/`). Un bug trovato = un test che lo riproduce.
4. **Metriche**: PR-AUC, ROC-AUC, precision@recall=0.9 e matrice di confusione a una soglia scelta sulla
   VALIDAZIONE (mai 0.5 a priori, mai sul test), sempre con la baseline (random forest sulle
   feature scalari) e con la dimensione del test set. Mai solo l'accuratezza.
5. **Nessun numero inventato.** I risultati stanno in `reports/*.json`; README e documenti li citano da
   li', dopo averli eseguiti. Se non hai eseguito qualcosa, dillo.
6. **Linguaggio scientifico onesto**: "candidato", mai "pianeta scoperto". Un classificatore sulla sola
   curva di luce non vede centroide, spettri, immagini: c'e' un tetto alle prestazioni, scrivilo.
7. **Ollama (LLM locale) solo per riformulare testo** a partire da numeri gia' calcolati. Mai per
   classificare o calcolare. Ogni testo passa da `llm.numbers_not_in_source`.
8. **Riproducibilita'**: seed espliciti, config e metriche salvate in JSON, versioni delle librerie nel report.
9. `scripts/download_kepler.py` e' stato verificato contro TAP/MAST reali sul set completo (6424 stelle).
   Difetti corretti: FITS da 0 byte nella cache di lightkurve e `sys.stdout` chiuso con i thread (si usano
   processi). I timeout di rete sono normali: rilancia lo stesso comando. Per nuovi errori usa la skill
   `diagnose`, con test per la logica pura. La rete non entra nei test.
10. La UI (`web/`) non contiene numeri scritti a mano: tutto da `demo.json`, generato da `export_web_data.py`.
    Gli avvisi "dati sintetici" vanno tolti solo quando esistono risultati su dati reali.

## Uso dei plugin
- **superpowers**: per funzionalita' non banali (brainstorming, piano, esecuzione con TDD). Per modifiche
  piccole non attivare il flusso completo.
- **mattpocock-skills**: `diagnose` per bug e metriche sospette *prima* di cambiare iperparametri a
  caso; `grill-with-docs` o `design-an-interface` per decisioni di design. Il TDD lo guida superpowers:
  non usare due flussi TDD insieme.
- **ponytail**: livello `lite` per script ed esplorazione; `off` per `preprocess`, `splits`, `metrics`,
  `llm` e per tutto cio' che tocca la correttezza dei dati o della valutazione.
- **code-simplifier**: a fine feature, sulle modifiche recenti, prima del commit; poi rileggi il diff.
- **frontend-design**: solo per la demo web (Fase 7).
- **github**: issue per fase, branch per feature, PR con descrizione dei risultati.
- **cowork-plugin-management**: non serve qui; ignoralo (vedi `docs/PLUGINS_AND_OLLAMA.md` per disattivarlo).

## Struttura
```
src/exotransit/   synthetic.py  preprocess.py  model.py  metrics.py  splits.py  koi.py  llm.py
scripts/          check_env.py  download_kepler.py  build_dataset.py  train.py  vetting_report.py
                  export_web_data.py  sanity_checks.py
web/              demo statica Vite+React per Vercel (legge web/public/data/demo.json)
tests/            un file di test per area; devono girare in pochi secondi, senza rete
docs/             ROADMAP.md  PROMPTS.md  SCIENCE_NOTES.md  PLUGINS_AND_OLLAMA.md
                  VERIFYING_AGENT_WORK.md  DEPLOY_VERCEL.md
data/ models/ reports/   artefatti locali (ignorati dal git, tranne .gitkeep)
```

## Convenzioni
- Tempo in giorni (BKJD per Kepler); durata del transito in **giorni** (la tabella KOI la da' in ore:
  la conversione e' in `koi.py`); profondita' come **frazione** del flusso (KOI: ppm).
- Viste: `GLOBAL_BINS=2001`, `LOCAL_BINS=201` in `preprocess.py`. Se li cambi, rigenera i dataset.
- Dataset `.npz`: `global_view (N,2001)`, `local_view (N,201)`, `extra (N,4)`, `y`, `groups` (kepid o
  star_id), `names`.
- Il detrending deve mascherare i transiti (vedi `preprocess.detrend`): non rimuovere la maschera.
- Commit piccoli e descrittivi; non committare `.env`, dati, pesi.

## Definizione di "fatto" per una modifica
Test passano, `scripts/sanity_checks.py` senza FAIL, `ruff check .` pulito, metriche rieseguite se hai toccato dati/modello, documentazione
e checkbox della roadmap aggiornate, nessun numero nel testo che non provenga da un report.
