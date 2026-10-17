# Roadmap

Tempo indefinito: l'ordine conta, le date no. Ogni fase ha un criterio di completamento
verificabile. Spunta le caselle solo quando il criterio e' davvero soddisfatto.

## Fase 0: Setup e verifica
- [x] Ambiente creato, `pip install -e ".[dev]"`, PyTorch con CUDA installato (README)
- [x] `python scripts/check_env.py` mostra la GPU e le librerie OK
- [x] `pytest` verde, `ruff check .` pulito
- [x] `git init`, primo commit, repo su GitHub, un'issue per fase (plugin `github`)
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
| Random forest, 4 feature scalari | 0.936 +- 0.006 | 0.859 +- 0.018 | 0.776 +- 0.017 |
| CNN completa (globale + locale + scalari) | 0.973 +- 0.007 | 0.949 +- 0.019 | 0.887 +- 0.020 |
| CNN senza feature scalari | 0.970 +- 0.006 | 0.948 +- 0.011 | 0.888 +- 0.020 |
| CNN solo vista locale | 0.963 +- 0.010 | 0.941 +- 0.014 | 0.844 +- 0.038 |
| CNN solo vista globale | 0.958 +- 0.006 | 0.928 +- 0.014 | 0.813 +- 0.032 |

Differenze appaiate (stesso split per seed):
- CNN completa - RF: PR-AUC +0.090 +- 0.026 (migliore in 5/5 seed); ROC-AUC +0.037; P@R90 +0.111.
  CNN senza scalari - RF: PR-AUC +0.089 +- 0.020 (5/5).
- Le due torri insieme battono ciascuna da sola: PR-AUC +0.021 +- 0.018 sulla globale (4/5 seed) e
  +0.008 +- 0.009 sulla locale (4/5); differenze dello stesso ordine della std, quindi modeste.
- Le feature scalari non aggiungono nulla di misurabile: senza di esse la PR-AUC cambia di -0.001 +- 0.009
  (2/5 seed meglio). Con 5 seed questo significa "nessuna differenza rilevata", non equivalenza dimostrata.
- **Riproducibilita'**: questi numeri vengono da una *seconda* esecuzione (con punteggi e pesi salvati) della
  stessa configurazione; la prima dava PR-AUC 0.951 (full) e 0.954 (no_scalar). La GPU non e' deterministica: le
  medie cambiano di 0.002-0.006 e il singolo seed fino a 0.02 (es. no_scalar seed 2: 0.951 -> 0.932). La random
  forest, deterministica, resta identica. Le conclusioni qualitative non cambiano.

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
- [x] Analisi errori: guarda a occhio falsi positivi e falsi negativi e raggruppali (periodo, profondita', SNR)
- [x] Calibrazione: reliability diagram, scegli la soglia per un recall obiettivo
- [x] Injection-recovery: inietta transiti sintetici in curve reali "tranquille" e misura la completezza
      in funzione di profondita' e periodo
- [x] Robustezza: cosa succede con meno trimestri di dati, con rumore aggiunto

**Fatto quando**: puoi dire in quali regimi il modello funziona e in quali no, con numeri.

**Stato**: soddisfatto. Modello di riferimento: CNN `no_scalar` (solo curve), 5 seed, soglia di ogni seed scelta
sulla validazione. Fonti: `reports/kepler_errors.json`, `kepler_injection.json`, `kepler_robustness.json`;
figure in `reports/figures/` (`errors/`, `calibration_reliability.png`, `injection_recovery.png`, `robustness.png`).

**Dove funziona e dove no** (test dei 5 seed messi insieme: 3563 righe, 2973 KOI distinti; la stessa stella puo'
comparire in piu' seed, quindi le righe per fascia non sono indipendenti):

| Periodo (d) | n | positivi | recall | precisione | tasso FP |
|---|---|---|---|---|---|
| <=1.5 | 606 | 34 | 0.47 | 0.53 | 0.02 |
| 1.5-3 | 481 | 105 | 0.76 | 0.79 | 0.06 |
| 3-10 | 844 | 436 | 0.92 | 0.90 | 0.11 |
| 10-30 | 774 | 454 | 0.93 | 0.92 | 0.12 |
| 30-100 | 439 | 222 | 0.92 | 0.94 | 0.06 |
| >100 | 419 | 63 | 0.86 | 0.70 | 0.07 |

| Profondita' (ppm) | n | positivi | recall | precisione | tasso FP |
|---|---|---|---|---|---|
| <=200 | 981 | 305 | 0.81 | 0.82 | 0.08 |
| 200-500 | 801 | 407 | 0.90 | 0.89 | 0.11 |
| 500-1000 | 543 | 347 | 0.94 | 0.94 | 0.11 |
| 1000-2500 | 353 | 178 | 0.94 | 0.90 | 0.10 |
| 2500-10000 | 213 | 49 | 0.98 | 0.87 | 0.04 |
| >10000 | 672 | 28 | 0.68 | 0.63 | 0.02 |

| SNR (catalogo) | n | positivi | recall | precisione | tasso FP |
|---|---|---|---|---|---|
| <=7.1 | 141 | 2 | 0.50 | 0.20 | 0.03 |
| 7.1-10 | 236 | 18 | 0.33 | 0.27 | 0.07 |
| 10-15 | 460 | 137 | 0.73 | 0.75 | 0.10 |
| 15-25 | 704 | 385 | 0.86 | 0.89 | 0.13 |
| 25-50 | 710 | 432 | 0.96 | 0.93 | 0.11 |
| 50-100 | 359 | 182 | 0.97 | 0.93 | 0.07 |
| 100-500 | 469 | 134 | 0.96 | 0.94 | 0.02 |
| >500 | 484 | 24 | 0.79 | 0.68 | 0.02 |

- **Regimi deboli**: periodi sotto ~3 d (pochi positivi: 34 sotto 1.5 d, recall 0.47); dip piu' profondi dell'1%
  (672 righe, solo 28 positivi: sono quasi tutte binarie a eclisse, e i pochi pianeti veri come K01450.01 vengono
  scartati); SNR sotto ~15 (con SNR<10 i positivi sono 20 su 377 righe: stime molto incerte).
- **Regime forte**: periodi 3-100 d, profondita' 500-10000 ppm, SNR 25-500: recall 0.92-0.98, precisione 0.87-0.94.
- **Errori guardati a occhio** (`reports/figures/errors_notes.md`, osservazioni soggettive): i falsi negativi sono in gran
  parte segnali deboli o curve corte/rumorose; i falsi positivi visti (5 su 12) hanno transiti puliti di forma
  planetaria, distinguibili solo con centroide/imaging: e' il tetto della sola curva di luce.

**Calibrazione** (isotonica adattata sulla validazione di ogni seed, test; media +- std sui 5 seed): ECE 0.055 +- 0.010
con il punteggio grezzo (addestrato con `pos_weight`, quindi non e' una probabilita') e 0.024 +- 0.007 dopo la
calibrazione isotonica; Brier 0.069 -> 0.062. Reliability: `reports/figures/calibration_reliability.png`.
**Soglia per recall obiettivo** scelta sulla validazione e applicata al test: obiettivo 0.90 -> recall 0.887 +- 0.022,
precisione 0.892 +- 0.019; obiettivo 0.95 -> recall 0.944 +- 0.009, precisione 0.812 +- 0.036. La soglia si trasferisce bene.

**Injection-recovery** (2520 iniezioni, 2506 valide con >=3 transiti osservati; 12 stelle per cella e per seed, circa 60
per cella; stelle di test del seed, meta' piu' tranquilla; transiti dei KOI veri eliminati). Completezza = quota con
punteggio >= soglia di validazione (media sui seed):

| profondita' \ periodo (d) | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| 200 ppm | 0.93 | 0.88 | 0.88 | 0.80 | 0.71 | 0.67 | 0.41 |
| 500 ppm | 0.98 | 0.97 | 0.95 | 0.88 | 0.85 | 0.87 | 0.83 |
| 1000 ppm | 1.00 | 0.98 | 0.98 | 0.98 | 0.92 | 0.93 | 0.85 |
| 2000 ppm | 1.00 | 1.00 | 1.00 | 0.97 | 0.98 | 0.93 | 0.92 |
| 5000 ppm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.97 |
| 10000 ppm | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |

La completezza cala con profondita' piccola e periodo lungo (meno transiti): a 200 ppm e P=128 d scende a 0.41.
Limiti: l'effemeride e' nota, quindi si misura il *classificatore* e non la rivelazione (niente BLS); le iniezioni sono
pianeti per costruzione (solo completezza, nessun tasso di falsi positivi); la durata segue P^(1/3) (stella tipo Sole,
transito centrale). La griglia parte da 2 d: i periodi < 1.5 d, dove il modello va peggio, non sono coperti.

**Robustezza** (KOI di test di ogni seed, soglia fissa dalla validazione pulita, media +- std sui 5 seed; confronto con
l'originale sullo stesso sottoinsieme; le viste originali ricostruite riproducono i punteggi salvati: PR-AUC 0.948,
recall 0.897):

| Perturbazione | KOI tenuti | PR-AUC | recall | precisione |
|---|---|---|---|---|
| originale | 713/713 | 0.948 +- 0.011 | 0.897 +- 0.012 | 0.883 |
| primi 730 d | 664/713 | 0.931 +- 0.019 (-0.020) | 0.793 +- 0.016 (-0.104) | 0.904 |
| primi 365 d | 636/713 | 0.898 +- 0.020 (-0.053) | 0.643 +- 0.036 (-0.254) | 0.912 |
| rumore x2 | 713/713 | 0.883 +- 0.027 (-0.065) | 0.623 +- 0.038 (-0.274) | 0.897 |
| rumore x4 | 713/713 | 0.741 +- 0.057 (-0.207) | 0.336 +- 0.022 (-0.560) | 0.870 |

Con meno dati o piu' rumore il modello diventa piu' prudente (la precisione resta ~0.87-0.91) ma perde recall: il
punteggio si abbassa e la soglia, tarata su dati puliti, non lo segue. Il ranking (PR-AUC) degrada molto meno del
recall a soglia fissa. Con 1 anno di dati invece di 4, il recall scende di 0.25. "Rumore xN" porta lo scatter
punto-a-punto a N volte l'originale con rumore gaussiano bianco: il rumore reale e' correlato e non e' coperto.

## Fase 5: TESS (opzionale)
- [x] Scarica TESS (tabelle TOI + curve di luce) e adatta il preprocessing (sector brevi, cadenza diversa)
      (**sottoinsieme**: 1476 stelle su 2657; vedi sotto)
- [x] Valuta il modello Kepler "as is" su TESS, poi fine-tuning
- [x] Documenta le differenze (meno transiti per candidato, sistematiche diverse)

**Fatto quando**: hai una tabella "Kepler -> TESS" onesta, anche se il trasferimento va male.

**Stato**: soddisfatto, ma su un sottoinsieme dei dati e con test piccoli (leggi i limiti).

**Dati** (`reports/tess_dataset.json`, `data/raw/tess_failed*.txt`): TOI con CP/KP = positivi e FP/FA = negativi (PC/APC
esclusi come i CANDIDATE Kepler): 2802 TOI su 2657 stelle. Scaricate 1476 stelle (SPOC 2 minuti, binnato a 10 minuti, ~150 MB);
168 stelle non hanno SPOC 2-min (solo FFI); 1013 non sono state scaricate perche' il download si e' fermato quando lo
spazio libero sul disco C: e' sceso sotto la soglia (spazio consumato da altro, non dal progetto). Le stelle non scaricate
sono quelle in coda all'ordine casuale (seed 0), quindi il sottoinsieme non e' scelto per classe ne' per qualita'.
`data/processed/tess.npz`: **1543 esempi da 1456 stelle** (836 positivi, 707 negativi; 64 stelle con piu' TOI); scartati 21 TOI
con meno di 2 transiti osservati (`--min-transits 2`: un settore dura ~27 giorni; Kepler usa 3).

**Trasferimento** (`reports/tess_transfer.json`, 5 seed; ogni seed ha il suo split per stella di TESS e usa il modello
Kepler `no_scalar` dello stesso seed; test di 148-158 esempi da 145 stelle; media +- std sui seed):

| Approccio | ROC-AUC | PR-AUC | P@R90 |
|---|---|---|---|
| Kepler **as is** (nessun riaddestramento) | 0.837 +- 0.014 | 0.840 +- 0.035 | 0.762 +- 0.018 |
| Kepler + **fine-tuning** su TESS (lr 1e-4, <=10 epoche) | 0.868 +- 0.035 | 0.866 +- 0.057 | 0.780 +- 0.028 |
| CNN **da zero** solo su TESS | 0.826 +- 0.034 | 0.855 +- 0.038 | 0.711 +- 0.037 |
| Random forest, 3 feature (profondita', periodo, durata/periodo) | 0.725 +- 0.027 | 0.742 +- 0.043 | 0.675 +- 0.012 |

- Il modello Kepler as is su **tutti** i 1543 TESS (soglia di Kepler): ROC-AUC 0.827-0.848 e PR-AUC 0.827-0.848 a seconda del
  seed, con IC bootstrap per stella larghi circa +-0.03 (es. seed 0: PR-AUC 0.832 [0.802, 0.861]); alla soglia di Kepler
  precisione 0.78-0.81 e recall 0.74-0.77 (prevalenza 0.54). Su Kepler, stessa classe di modello: ROC-AUC 0.970, recall 0.897
  e precisione 0.883: **il trasferimento perde parecchio** (la PR-AUC non e' confrontabile: la prevalenza e' 0.54 contro 0.35).
- Differenze appaiate: Kepler as is - random forest: PR-AUC +0.098 +- 0.064 (5/5 seed). Fine-tuning - as is: PR-AUC
  +0.026 +- 0.033 (4/5 seed), ROC-AUC +0.031 +- 0.023 (5/5): piccolo e dello stesso ordine della std. Fine-tuning - da zero:
  PR-AUC +0.012 +- 0.063 (2/5): **non distinguibili**. As is - da zero: -0.015 +- 0.045 (1/5): nemmeno.
  Quindi il modello addestrato su Kepler, senza vedere TESS, e' tanto buono quanto uno addestrato su ~1300 esempi TESS.
- Controllo con etichette TESS permutate tra stelle (`reports/tess_transfer_shuffled.json`, 2 seed, test di ~150 esempi):
  ROC-AUC 0.55-0.57 per le CNN e 0.50 per la random forest (rumore del test piccolo: +-0.06); Kepler as is su tutti i TESS con
  etichette permutate: 0.501. Nessun leakage evidente.
- **Limiti**: sottoinsieme di TESS (1543 di ~2800 TOI); test di soli ~150 esempi per seed (le 5 stime di split condividono molte
  stelle), quindi le differenze fra approcci sono poco potenti; un solo set di iperparametri di fine-tuning, non ottimizzato;
  il catalogo TOI da' un'epoca di riferimento e per dati lontani l'errore sul periodo si accumula (non misurato qui);
  nessuna analisi per fasce di periodo/SNR su TESS (l'SNR non c'e'); etichette TFOP.

**Differenze Kepler / TESS che contano**: cadenza 2 minuti (qui 10) contro 30; pochi transiti per candidato (un settore dura
27 giorni: le viste globali hanno meno punti e sono piu' rumorose, e per questo si accettano 2 transiti invece di 3);
sistematiche e stelle diverse (TESS osserva stelle piu' brillanti e vicine, con pixel grandi: piu' contaminazione da
stelle vicine); etichette dalla comunita' (TFOP) e non dal team Kepler; nessuna colonna SNR nel catalogo TOI, quindi si usa
il modello solo-curve.

## Fase 6: Schede di vetting con Ollama
- [x] Scegli 2 modelli locali e confrontali su ~50 candidati con `llm.vetting_report`
- [x] Misura: quota di report con numeri sospetti (`suspicious_numbers`), uso del fallback, tempo
- [x] Decidi quale modello usare e documenta i limiti

**Fatto quando**: la demo non mostra mai un numero che non provenga dalla pipeline.

**Stato**: soddisfatto con una condizione (leggi sotto). Fonti: `reports/vetting_llm_comparison.json` (prompt v1),
`reports/vetting_llm_comparison_v2.json` (prompt v2), revisione a mano in `reports/vetting_llm_review.md`.

**Modelli** (GPU da 4 GB: entrambi con parziale offload su CPU; un modello alla volta): `qwen2.5-coder:7b` (4.7 GB, veloce,
ma e' un modello per codice) e `gemma3:12b` (8.1 GB, multilingue). Scartati: `deepseek-r1:8b` (modello che "ragiona",
produce testo di pensiero), `phi4:14b` e `deepseek-coder-v2:16b` (troppo grandi per 4 GB). 50 candidati a caso dal test del
seed 0 Kepler (mai visti dal classificatore), numeri dalla pipeline con probabilita' calibrata (isotonica sulla
validazione); temperatura 0.1; una chiamata di riscaldamento esclusa dai tempi.

| Prompt | Modello | Fallback | Flag automatici | Tempo mediano / p95 | Parole | Mia revisione a mano |
|---|---|---|---|---|---|---|
| v1 (JSON) | qwen2.5-coder:7b | 0% | 0% col controllo di partenza, 42% con quelli nuovi | 6 s / 10 s | 35 | 19/50 SNR confrontato con la soglia; 2 "e' un esoplaneta" |
| v1 (JSON) | gemma3:12b | 0% | 0% col controllo di partenza, 98% con quelli nuovi | 30 s / 33 s | 64 | ~18/50 durata scambiata col periodo; 47/50 raccomandazioni non supportate |
| v2 (righe etichettate) | qwen2.5-coder:7b | 0% | 2% | 12 s / 15 s | 60 | >=22/50 interpretazioni astrofisiche non supportate o false |
| v2 (righe etichettate) | gemma3:12b | 0% | 0% | 33 s / 36 s | 65 | **0/50 errori rilevati** |

- **Il controllo di partenza (numeri non presenti nei dati) non basta**: i difetti reali sono *numeri validi nel ruolo
  sbagliato*, che non vede. Ho aggiunto a `check_report` controlli per SNR/soglia, durata/periodo, raccomandazioni e
  linguaggio da scoperta (test con i testi reali). Sono euristiche tarate sui testi osservati: la v2 e' stata valutata su un
  campione nuovo proprio per questo, ma segnalano solo 1 delle ~22 schede difettose di qwen v2: **la lettura a mano e'
  insostituibile**.
- **Difetti del nostro codice trovati durante il confronto** (corretti, con test): il nome del candidato segnalato come numero
  inventato; "93%" per 0.93 segnalato; chiave `snr_bls` che faceva dire "SNR BLS" per un SNR del catalogo; prompt che
  invitava a scrivere "non disponibile" e faceva inventare campi.
- **Decisione**: usare `gemma3:12b` con il prompt v2, *precalcolato* (non c'e' LLM in tempo reale: ~33 s a scheda), con ogni
  testo che passa da `check_report` e ricade sul template deterministico se segnalato o se Ollama non risponde. Scartare
  `qwen2.5-coder:7b`.
- **Limiti**: 50 schede per modello e una sola persona/agente che le legge (0/50 errori vuol dire "nessuno visto", non zero;
  con 50 campioni il limite superiore al 95% e' circa 7%). Le schede di gemma v2 sono quasi identiche tra loro: il modello
  fa da parafrasatore e il valore rispetto al template e' basso. La calibrazione isotonica da' valori esatti 0.0 e 1.0
  ("probabilita' del 100%"): nella demo va mostrato come punteggio calibrato approssimato. Un solo seed del classificatore e un
  solo prompt v2 provato (iterato su un campione, valutato su un altro).

## Fase 7: Demo web (opzionale)
- [x] Demo statica per Vercel in `web/` (risultati precalcolati sintetici; vedi `docs/DEPLOY_VERCEL.md`)
- [x] Riesportare i dati della demo con risultati Kepler reali e aggiornare i testi di avviso
- [ ] (opzionale) Inferenza live: ONNX nel browser o API separata (**non fatta, scelta deliberata**, vedi `docs/DEPLOY_VERCEL.md`)
- [x] Interfaccia (skill `frontend-design`): grafico della curva, curva ripiegata, probabilita', scheda
- [x] Avviso visibile: "punteggio di un classificatore, non una scoperta"

**Fatto quando**: un recruiter puo' provarla in due minuti con un esempio precaricato.

**Stato**: soddisfatto in locale; **il deploy su Vercel non e' stato fatto** (serve il tuo account, istruzioni in
`docs/DEPLOY_VERCEL.md`). Sezioni della pagina: 60 candidati Kepler del test del seed 0 (18 TP, 18 TN, 12 FP, 12 FN scelti a caso
per esito: gli errori sono sovra-rappresentati e la pagina lo dice) con curva di luce, transiti previsti, viste globale e locale,
punteggio, probabilita' calibrata, etichetta dell'archivio e scheda di testo (gemma3:12b, tutte e 60 passano `check_report`; la
pagina dichiara che il testo puo' contenere errori); "modalita' indovina" che nasconde la risposta; poi i risultati (baseline, fasce,
injection, robustezza, calibrazione, TESS, schede) copiati dai report; limiti.
- **Dati**: `web/public/data/demo.json` (~3.5 MB), generato da `export_web_data.py` (si ferma se i punteggi salvati non coincidono col
  modello) e controllato da `check_demo.py` (13 controlli indipendenti, tutti passati: punteggi ricalcolati dal modello, esempi
  solo dal test, blocchi dei risultati = report, testi LLM senza difetti noti).
- **Verifiche fatte**: `tsc --noEmit` e `npm run build` puliti; anteprima in Chrome headless in tema scuro. **Non verificati**: tema
  chiaro (il flag per forzarlo non ha funzionato), layout sotto ~500 px di larghezza (Chrome headless non scende oltre), accessibilita'
  oltre focus e contrasti di base, comportamento senza rete (i font vengono da Google Fonts, con fallback di sistema).
- **Regola 10 (nessun numero scritto a mano nella UI)**: controllata con una grep sul testo JSX: l'unico residuo e' il concetto
  "0 e 1 esatti" della calibrazione isotonica, non un risultato.

## Fase 8: Portfolio
- [x] README con risultati copiati da `reports/` (nessun numero a memoria), figure, limiti
- [x] Model card: dati, uso previsto, metriche, limiti, rischi
- [ ] CI su GitHub Actions: `pytest` + `ruff` (**scritta, non ancora eseguita su GitHub**: spuntala quando il primo run e' verde)
- [x] Post/riassunto di una pagina: problema, metodo, risultati onesti, cosa faresti dopo
- [x] Citazioni verificate (vedi `SCIENCE_NOTES.md`)
- [ ] Scegli una licenza (decisione tua: oggi il repository, che e' pubblico, non ne ha nessuna)

**Fatto quando**: un estraneo capisce cosa hai fatto, lo riproduce e vede dove il progetto non arriva.

**Stato**: documenti pronti. README, `docs/MODEL_CARD.md` e `docs/POST.md` sono **generati** da `docs/templates/*.tmpl` con
`scripts/render_docs.py` a partire dagli snapshot in `docs/results/` (creati da `scripts/snapshot_results.py`, i report completi
sono ignorati dal git): nessun numero e' scritto a mano e `render_docs.py --check` verifica che siano aggiornati (anche in CI).
Le citazioni sono state verificate su Crossref (DOI, titolo, anno, rivista, volume). Non verificato: l'esecuzione della CI su
GitHub (workflow non provato su Linux), l'aspetto del README su GitHub. Il repository e' pubblico (HTTP 200 anonimo) e la
scansione di segreti non ha trovato nulla.
