# Roadmap

Tempo indefinito: l'ordine conta, le date no. Ogni fase ha un criterio di completamento
verificabile. Spunta le caselle solo quando il criterio e' davvero soddisfatto.

## Fase 0: Setup e verifica
- [x] Ambiente creato, `pip install -e ".[dev]"`, PyTorch con CUDA installato (README)
- [x] `python scripts/check_env.py` mostra la GPU e le librerie OK
- [x] `pytest` verde, `ruff check .` pulito
- [ ] `git init`, primo commit, repo su GitHub, un'issue per fase (plugin `github`)
- [x] Smoke test sintetico eseguito (comandi in README) e risultato salvato in `reports/`
      (`reports/synthetic_metrics.json`: GPU, split 240/30/30, test di soli 30 esempi)

**Fatto quando**: i comandi di Quickstart girano da zero su una macchina pulita.

## Fase 1: Dati reali Kepler
- [x] `scripts/download_kepler.py --limit 50` funziona (corretto con `diagnose`: cache FITS troncata, thread -> processi;
      50/50 stelle, 0 fallite)
- [x] Conteggi per classe stampati e annotati (CONFIRMED vs FALSE POSITIVE)
      (output di `download_kepler.py` su `data/raw/koi.csv`, dopo i filtri di `prepare_koi_table`:
      2746 CONFIRMED, 4582 FALSE POSITIVE; 1977 CANDIDATE esclusi; smoke test: `reports/kepler_smoke.json`)
- [x] Ispezione visiva di ~20 curve: serie temporale, curva ripiegata, vista globale e locale
      (20 stelle, 10+10; note soggettive in `reports/figures/inspection_notes.md`, nessuna anomalia sistematica)
- [x] Sanity check: un KOI CONFIRMED con SNR alto mostra un transito allineato al centro della vista locale
      (KIC 3642289, K00301.01, controllato a occhio in `reports/figures/kepid_3642289.png`; una sola stella)
- [x] Test sulla logica pura che hai dovuto correggere (`tests/test_download.py`)
- [x] Scarica il set completo (ore e molti GB: controlla lo spazio disco prima)
      (6424/6424 stelle, circa 8 ore con 8 processi (15:22-23:17); 50 timeout di rete al primo passaggio, 0 fallite dopo il secondo;
      `data/raw/kepler` 2950 MB)

**Fatto quando**: `data/processed/kepler.npz` esiste, la distribuzione di classi e' documentata e le
viste di esempio sono state controllate a occhio.

**Stato**: soddisfatto. `data/processed/kepler.npz`: 7181 esempi da 6294 stelle (2709 positivi, 4472 falsi positivi;
601 stelle con piu' KOI), 147 KOI scartati (146 con <3 transiti osservati, 1 errore). Fonte: `reports/kepler_dataset.json`.
Nota: i KOI con SNR<7.1 sono 4 positivi contro 270 negativi (dal dataset), ma la relazione SNR-classe non e'
monotona (vedi Fase 2): le metriche per fascia di SNR restano essenziali.

## Fase 2: Baseline
- [x] Soglia sul solo SNR (punteggio = SNR del catalogo KOI; non e' un SNR BLS: le effemeridi sono quelle NASA)
- [x] Random forest sulle 4 feature scalari (`src/exotransit/baselines.py`, `scripts/baselines.py`)
- [x] Tabella baseline con ROC-AUC, PR-AUC, precision@recall=0.9 e dimensione del test set

**Fatto quando**: hai il numero da battere, con intervalli di incertezza (bootstrap sul test set).

**Stato**: soddisfatto. Fonte: `reports/kepler_baselines.json` (seed 0, split per stella come `train.py`, test di
745 esempi da 631 stelle con 322 positivi; IC 95% con bootstrap per stella, 1000 repliche; soglia dalla validazione).

| Modello | ROC-AUC | PR-AUC | P@R90 |
|---|---|---|---|
| Solo SNR (soglia) | 0.491 [0.447, 0.534] | 0.385 [0.342, 0.433] | 0.482 [0.439, 0.532] |
| Random forest, solo SNR | 0.709 [0.674, 0.746] | 0.612 [0.549, 0.677] | 0.541 [0.471, 0.599] |
| **Random forest, 4 feature scalari** | **0.926 [0.906, 0.942]** | **0.888 [0.848, 0.920]** | **0.789 [0.725, 0.843]** |

Letture, e limiti:
- L'SNR da solo, come punteggio monotono, non separa le classi (ROC-AUC ~0.49, verificato anche sulla colonna
  grezza del catalogo: 0.475). Non e' un bug: la frazione di positivi e' ~2% sotto SNR 7.1, ~60% tra 25 e 50 e
  ~7% sopra 500 (binarie a eclisse molto profonde). Un random forest sul solo SNR cattura parte di questa forma.
- Le 4 feature sono ricavate dal catalogo KOI (effemeride, profondita', durata, SNR della pipeline Kepler), non da
  una misura fatta sulle nostre curve: la baseline e' forte proprio perche' contiene il lavoro del team Kepler.
  La CNN dovra' battere 0.926 / 0.888, e vedere solo la curva ha comunque un tetto (niente centroide, spettri).
- Un solo split (seed 0): l'IC bootstrap misura il campionamento del test, non la variabilita' tra split.

## Fase 3: CNN su Kepler
- [x] Addestramento con split per stella, seed fissati, curve di apprendimento salvate
      (`reports/figures/cnn_learning_curves.png`; la storia per epoca e' in `reports/kepler_cnn.json`)
- [x] Almeno 5 seed: riporta media e deviazione standard, non il migliore
- [x] Ablation: solo vista globale, solo locale, senza feature scalari
- [x] Confronto con la baseline della Fase 2

**Stato**: soddisfatto per quanto riguarda il confronto; la validazione robusta e' la Fase 4.
Fonte: `reports/kepler_cnn.json` (5 seed 0-4; ogni seed ha il suo split per stella e su quello split si addestrano
CNN, ablation e random forest; test di 685-745 esempi per seed; iperparametri fissi di `train.py`, nessun tuning;
GPU, torch 2.14.1). Media +- std (ddof=1) sui seed:

| Modello | ROC-AUC | PR-AUC | P@R90 |
|---|---|---|---|
| Random forest, 4 feature scalari | 0.936 +- 0.006 | 0.859 +- 0.018 | 0.776 +- 0.017 |
| CNN completa (globale + locale + scalari) | 0.972 +- 0.008 | 0.951 +- 0.014 | 0.888 +- 0.035 |
| CNN senza feature scalari | 0.973 +- 0.004 | 0.954 +- 0.012 | 0.913 +- 0.011 |
| CNN solo vista locale | 0.961 +- 0.013 | 0.938 +- 0.020 | 0.825 +- 0.049 |
| CNN solo vista globale | 0.957 +- 0.010 | 0.928 +- 0.017 | 0.817 +- 0.052 |

Differenze appaiate (stesso split per seed):
- CNN completa - RF: PR-AUC +0.091 +- 0.022 (migliore in 5/5 seed); ROC-AUC +0.036; P@R90 +0.112.
- Le due torri insieme battono ciascuna da sola: PR-AUC +0.022 +- 0.019 sulla globale (4/5 seed) e
  +0.012 +- 0.011 sulla locale (5/5); differenze dello stesso ordine della std, quindi modeste.
- Le feature scalari non aggiungono nulla di misurabile: senza di esse la CNN fa PR-AUC +0.004 +- 0.006 (3/5 seed
  meglio). Con 5 seed questo significa "nessuna differenza rilevata", non una dimostrazione di equivalenza.

Controlli e limiti:
- Controllo con etichette permutate tra stelle (`--shuffle-labels`, seed 0-1, `reports/kepler_cnn_shuffled.json`):
  ROC-AUC 0.479 +- 0.028 (full), 0.487 +- 0.020 (no_scalar), 0.494 +- 0.019 (RF): tutti ~0.5, nessun leakage
  evidente nella pipeline di split e viste (il test e' su 2 seed soltanto).
- Non esclude scorciatoie nel segnale vero: le viste usano l'effemeride del catalogo e la forma del dip rivela
  le binarie a eclisse (eclisse secondaria, V). Il tetto della sola curva di luce resta (niente centroide, spettri).
- La validazione (~700 esempi) e' rumorosa: la loss di validazione oscilla e l'early stopping sceglie epoche tra 2 e
  13. La CNN va in sovra-adattamento dopo poche epoche (loss di training in calo, di validazione no).
- Non confrontabile con la letteratura (altre etichette e altro campione): qui i FALSE POSITIVE sono le disposizioni KOI.

**Fatto quando**: sai quale componente porta quanto, e hai un risultato difendibile (anche se modesto).
Se la CNN non batte la baseline, e' un risultato: analizzalo con `diagnose`, non nasconderlo.

## Fase 4: Valutazione robusta
- [ ] Analisi errori: guarda a occhio falsi positivi e falsi negativi e raggruppali (periodo, profondita', SNR)
- [ ] Calibrazione: reliability diagram, scegli la soglia per un recall obiettivo
- [ ] Injection-recovery: inietta transiti sintetici in curve reali "tranquille" e misura la completezza
      in funzione di profondita' e periodo
- [ ] Robustezza: cosa succede con meno trimestri di dati, con rumore aggiunto

**Fatto quando**: puoi dire in quali regimi il modello funziona e in quali no, con numeri.

## Fase 5: TESS (opzionale)
- [ ] Scarica TESS (tabelle TOI + curve di luce) e adatta il preprocessing (sector brevi, cadenza diversa)
- [ ] Valuta il modello Kepler "as is" su TESS, poi fine-tuning
- [ ] Documenta le differenze (meno transiti per candidato, sistematiche diverse)

**Fatto quando**: hai una tabella "Kepler -> TESS" onesta, anche se il trasferimento va male.

## Fase 6: Schede di vetting con Ollama
- [ ] Scegli 2 modelli locali e confrontali su ~50 candidati con `llm.vetting_report`
- [ ] Misura: quota di report con numeri sospetti (`suspicious_numbers`), uso del fallback, tempo
- [ ] Decidi quale modello usare e documenta i limiti

**Fatto quando**: la demo non mostra mai un numero che non provenga dalla pipeline.

## Fase 7: Demo web (opzionale)
- [x] Demo statica per Vercel in `web/` (risultati precalcolati sintetici; vedi `docs/DEPLOY_VERCEL.md`)
- [ ] Riesportare i dati della demo con risultati Kepler reali e aggiornare i testi di avviso
- [ ] (opzionale) Inferenza live: ONNX nel browser o API separata
- [ ] Interfaccia (skill `frontend-design`): grafico della curva, curva ripiegata, probabilita', scheda
- [ ] Avviso visibile: "punteggio di un classificatore, non una scoperta"

**Fatto quando**: un recruiter puo' provarla in due minuti con un esempio precaricato.

## Fase 8: Portfolio
- [ ] README con risultati copiati da `reports/` (nessun numero a memoria), figure, limiti
- [ ] Model card: dati, uso previsto, metriche, limiti, rischi
- [ ] CI su GitHub Actions: `pytest` + `ruff`
- [ ] Post/riassunto di una pagina: problema, metodo, risultati onesti, cosa faresti dopo
- [ ] Citazioni verificate (vedi `SCIENCE_NOTES.md`)

**Fatto quando**: un estraneo capisce cosa hai fatto, lo riproduce e vede dove il progetto non arriva.
