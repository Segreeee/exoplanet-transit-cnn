# Come verificare che il lavoro degli agenti funzioni davvero

Regola di base: **"fatto" detto da un agente e' un'affermazione, non una prova.** Una prova e' un
output che hai riprodotto tu, da uno stato pulito, e che puoi ricondurre a un file.
Gli agenti sbagliano in modo plausibile: codice che gira ma calcola la cosa sbagliata, test che
ripetono il codice, numeri citati a memoria. Qui sotto, in ordine di costo crescente.

## 1. Riproduci da zero (5 minuti, ogni fase)
```
git clone <repo> /tmp/check && cd /tmp/check
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # + PyTorch come da README
pytest && ruff check .
```
Se i comandi del README non girano su una macchina pulita, il lavoro non e' finito, qualunque cosa
dica il riassunto dell'agente. Controlla anche `git status`: file generati a mano, dati nel repo?

## 2. I test testano davvero qualcosa?
- **Sabotaggio**: rompi apposta il codice (es. togli la maschera dal detrending, scambia `<` e `>` in
  `precision_at_recall`) e verifica che almeno un test diventi rosso. Se resta verde, quel codice non e' protetto.
- **Rosso prima del verde**: per ogni bug chiedi un test che fallisca *prima* della correzione e
  guardalo fallire.
- Diffida di test che confrontano il risultato con una copia della stessa formula, e di `assert True`,
  tolleranze larghe, `skip`, `xfail`, `try/except: pass`.

## 3. Ogni numero in un file, ogni file riproducibile
- README e documenti citano solo numeri presenti in `reports/*.json` (regola in `CLAUDE.md`).
- Rigenera il report e confronta: stessi seed devono dare gli stessi numeri (qui: ROC-AUC 0.9375 uguale
  a due esecuzioni diverse).
- `python scripts/sanity_checks.py ...` ricalcola le metriche con sklearn dal modello salvato, controlla
  i duplicati tra split e la coerenza della demo web. Estendilo quando aggiungi risultati.

## 4. Controlli specifici del machine learning
| Controllo | Cosa smaschera | Nel progetto |
|---|---|---|
| Split per gruppo e verifica con insiemi | leakage tra esempi della stessa stella | `sanity_checks.py` |
| **Label shuffle** (etichette di training mescolate) | leakage o valutazione rotta: il test deve crollare al caso | `sanity_checks.py` |
| Baseline semplice | modello complesso che non aggiunge nulla | random forest, poi SNR da solo |
| Una feature alla volta | scorciatoie (es. "SNR alto = pianeta") | riga INFO di `sanity_checks.py` |
| Piu' seed, media e deviazione | risultato fortunato | Fase 3 della roadmap |
| Intervallo di confidenza (bootstrap) | differenze dovute al rumore con test set piccoli | Fase 2 |
| Soglia scelta sulla validazione | metriche "ottimiste" o recall zero a soglia 0.5 | `train.py` |
| Guardare a occhio gli errori | bug di dati che le metriche non mostrano | esploratore della UI, Fase 4 |
| Test fuori distribuzione / iniezione-recupero | modello che impara le peculiarita' del generatore | Fase 4 |

Un modello con ottime metriche e' *piu'* sospetto di uno con metriche modeste, finche' non hai passato
questi controlli.

## 5. Leggi il diff, non il riassunto
Dopo ogni sessione: `git diff --stat`, poi leggi i file toccati. Cerca:
- test cancellati, indeboliti o disattivati; soglie o tolleranze cambiate;
- numeri scritti a mano (nel codice, nel README, nella UI) invece che letti da un file;
- gestione errori che nasconde fallimenti; valori di default che mascherano dati mancanti;
- modifiche a `splits.py`, `metrics.py`, `preprocess.py` non richieste.
Il flusso `superpowers` con piano approvato prima del codice aiuta: confronti il diff col piano.

## 6. Un secondo agente come revisore, con cautela
Un agente "freddo" (senza il contesto di chi ha scritto il codice) con il compito esplicito di
*rompere* il lavoro (leakage, scorciatoie, test finti) trova cose. Ma condivide gli stessi punti ciechi
del primo: non sostituisce i punti 1-5, li integra.

## 7. Chiedi cosa NON e' stato verificato
Fine di ogni fase: "elenca cio' che non hai eseguito o verificato". Qui, a oggi: download Kepler reale,
GPU, Ollama con un modello vero, tutto cio' che e' su dati reali.

## Esempio reale da questo progetto
- La CNN sintetica sembrava ottima (ROC-AUC 0.94), ma a soglia 0.5 non riconosceva nessun pianeta
  (recall 0 su 14). Metriche di ranking buone, punteggi non calibrati: l'ha rivelato guardare la matrice
  di confusione e le singole predizioni, non l'AUC. Soluzione: soglia scelta sulla validazione.
- Dopo la correzione, sul test (30 esempi) la baseline random forest classifica quasi quanto la CNN:
  "la CNN batte la baseline" reggeva solo sul ranking e senza significativita'. Cio' che un riassunto
  entusiasta avrebbe lasciato fuori.
- Anche gli script di controllo sbagliano: la prima versione di `sanity_checks.py` usava gli id degli esempi
  come chiavi, ma gli id si ripetono (due candidati della stessa stella) e il controllo ha dato errore.
  L'hai visto solo perche' lo script e' stato eseguito.

## Cosa questi controlli NON dimostrano
Che il modello funzioni sul cielo vero. Quello richiede dati Kepler reali, split non leaky, baseline,
incertezze e, idealmente, qualcuno con competenza astrofisica che guardi gli errori. I controlli sopra
escludono una classe di errori (software e valutazione), non la validita' scientifica.
