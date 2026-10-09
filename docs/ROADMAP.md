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
Nota: i KOI con SNR<7.1 sono 4 positivi contro 270 negativi (dal dataset): l'SNR da solo e' gia' un forte
predittore, quindi la baseline della Fase 2 e le metriche per fascia di SNR sono essenziali.

## Fase 2: Baseline
- [ ] Soglia sul solo SNR BLS (punteggio = SNR)
- [ ] Random forest sulle 4 feature scalari (gia' in `scripts/train.py`)
- [ ] Tabella baseline con ROC-AUC, PR-AUC, precision@recall=0.9 e dimensione del test set

**Fatto quando**: hai il numero da battere, con intervalli di incertezza (bootstrap sul test set).

## Fase 3: CNN su Kepler
- [ ] Addestramento con split per stella, seed fissati, curve di apprendimento salvate
- [ ] Almeno 5 seed: riporta media e deviazione standard, non il migliore
- [ ] Ablation: solo vista globale, solo locale, senza feature scalari
- [ ] Confronto con la baseline della Fase 2

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
