# Prompt pronti per Claude Code

Apri Claude Code nella cartella del progetto (legge `CLAUDE.md` da solo) e incolla il prompt della
fase. Adatta i nomi delle skill a quello che vedi con `/help` o `/plugin`: non ho potuto verificarli
sulla tua installazione.

## Fase 0: verifica ambiente
```
Leggi CLAUDE.md e docs/ROADMAP.md. Poi esegui scripts/check_env.py, pytest e ruff check .
Riportami solo cio' che fallisce e proponi la correzione minima. Non cambiare codice senza dirmelo.
```

## Fase 1: dati reali (bug probabili)
```
Usa la skill diagnose. scripts/download_kepler.py non e' mai stato eseguito contro i server reali.
Eseguilo con --limit 5, osserva gli errori reali, formula ipotesi, correggi una cosa alla volta.
Per ogni correzione di logica pura aggiungi prima un test (niente rete nei test).
Alla fine mostrami: conteggi per classe, e per 3 stelle il grafico della curva grezza, appiattita,
ripiegata e le viste globale/locale (salva i PNG in reports/figures/).
```

```
Con superpowers: brainstorming + piano per "build_dataset --source kepler" sui dati scaricati.
Punti da decidere con me: gestione stelle con piu' KOI, curve con troppi buchi, scarto dei candidati con
SNR basso. Non scrivere codice prima che io approvi il piano.
```

## Fase 2: baseline
```
Aggiungi in scripts/train.py due baseline: (1) punteggio = SNR BLS, (2) random forest (gia' presente).
Test prima. Intervalli di confidenza al 95% con bootstrap sul test set. Salva tutto in reports/*.json.
Non toccare preprocess.py, splits.py, metrics.py senza mostrarmi il diff (ponytail: off).
```

## Fase 3: CNN e ablation
```
Piano con superpowers per: 5 seed, ablation (solo globale, solo locale, senza scalari), tabella media±std.
Esegui su GPU se disponibile. Prima di cambiare iperparametri su risultati strani usa diagnose.
Riporta solo numeri letti dai file in reports/. A fine lavoro code-simplifier sul diff, poi ruff + pytest.
```

## Fase 4: valutazione robusta
```
Usa design-an-interface per proporre 2 interfacce per l'injection-recovery (funzione che inietta un
transito sintetico in una curva reale e misura la completezza per profondita'/periodo).
Fammi scegliere, poi implementa con TDD. Includi calibrazione (reliability diagram) e analisi errori
con una griglia di falsi positivi/negativi da guardare a occhio.
```

## Fase 5: TESS (opzionale)
```
grill-with-docs: interrogami sulle differenze Kepler/TESS prima di progettare il preprocessing
(cadenza, lunghezza dei settori, tabella TOI). Poi piano e implementazione con superpowers.
```

## Fase 6: schede con Ollama
```
Esegui scripts/check_env.py per vedere i modelli Ollama installati. Scrivi uno script che confronta i
2 modelli che scelgo su 50 candidati con llm.vetting_report e misura: quota di suspicious_numbers,
fallback, tempo medio. Non cambiare SYSTEM_PROMPT senza dirmelo. Salva i risultati in reports/.
```

## Fase 7: demo web (opzionale)
```
Con superpowers pianifica una demo: FastAPI + pagina singola. Usa la skill frontend-design per
l'interfaccia. Vincoli: avviso "punteggio di un classificatore, non una scoperta"; esempio
precaricato; nessun numero nella scheda che non provenga dalla pipeline.
```

## Fase 8: portfolio e GitHub
```
Con il plugin github: crea una issue per ogni fase non completata della roadmap, apri una PR per il
lavoro corrente con descrizione che cita i numeri da reports/. Scrivi la model card in docs/MODEL_CARD.md
(dati, uso previsto, metriche con incertezza, limiti). Aggiungi una GitHub Action con pytest e ruff.
```

## Prompt di controllo (usali spesso)
```
Prima di dirmi che e' fatto: esegui pytest e ruff, riesegui la metrica toccata, e confronta ogni numero
che citi con il file in reports/. Elenca cio' che NON hai verificato.
```

```
Cerca leakage: dimmi dove vengono creati gli split, se piu' esempi della stessa stella possono finire
in split diversi, e se qualche statistica (normalizzazione, soglia) usa dati del test set.
```
