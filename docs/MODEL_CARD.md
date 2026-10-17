<!-- GENERATO da docs/templates/MODEL_CARD.md.tmpl con scripts/render_docs.py a partire da docs/results/*.json.
     Non modificare a mano: modifica il modello e rigenera. -->
# Model card: classificatore di candidati esopianeta da curve di luce

## Dettagli del modello
- **Cosa fa**: dato un candidato di transito con periodo, epoca e durata noti (dal catalogo), restituisce un punteggio che
  stima quanto la curva assomigli a un pianeta e non a un falso positivo.
- **Architettura**: CNN 1D a due torri (vista globale e vista locale della curva ripiegata sul periodo, binning a mediana),
  401.377 parametri nella variante di riferimento *solo curve*; esistono varianti con le feature del catalogo, solo
  vista locale e solo vista globale (confronto sotto). Ispirata ad AstroNet (Shallue & Vanderburg 2018), riscritta da zero:
  **non** è una riproduzione dei suoi risultati.
- **Software**: PyTorch 2.14.1+cu126, addestrata su GPU (cuda). Codice, test e istruzioni nel repository.
- **Sviluppatore**: progetto personale di portfolio. Nessuna licenza ancora scelta.

## Uso previsto e fuori ambito
- **Previsto**: studio, dimostrazione e confronto di metodi di *vetting* su curve di luce di Kepler (e, con cautela, TESS); priorità
  di verifica di candidati già noti.
- **Fuori ambito**: annunciare o confermare pianeti; sostituire il vetting con centroide, spettri o immagini; usarlo come
  rivelatore di transiti (il periodo viene dal catalogo, il modello non lo cerca); stelle o strumenti molto diversi da Kepler e TESS.
- **Linguaggio**: l'esito è "candidato" o "scartato", mai "pianeta scoperto".

## Dati
- **Kepler**: tabella `cumulative` dei KOI della NASA Exoplanet Archive; curve PDCSAP di tutti i trimestri (MAST via `lightkurve`).
  Etichetta 1 = CONFIRMED, 0 = FALSE POSITIVE; i CANDIDATE sono esclusi. 7181 esempi da 6294 stelle
  (2709 positivi, 4472 negativi; 601 stelle con più KOI). Scartati i KOI con meno di 3 transiti
  osservati. I transiti degli altri KOI della stessa stella sono eliminati prima del ripiegamento.
- **TESS**: tabella TOI; CP/KP = 1, FP/FA = 0, PC/APC esclusi; curve SPOC a 2 minuti binnate a 10 minuti. 1543
  esempi da 1456 stelle (836 positivi, 707 negativi): **sottoinsieme**, il download si è fermato quando il
  disco si è riempito. Minimo 2 transiti osservati (un settore dura circa 27 giorni).
- Le etichette sono disposizioni di gruppi di ricerca (Kepler) e della comunità (TFOP per TESS): non sono verità di laboratorio.

## Addestramento
Perdita BCE con `pos_weight` per lo sbilanciamento, AdamW (lr 0.001), batch 64, al massimo 30 epoche con
arresto anticipato sulla PR-AUC di validazione. Split per stella (circa 80% addestramento, 10% validazione, 10% test) diverso per
ciascuno dei 5 seed. La soglia di decisione si sceglie sulla validazione (F1 massimo). Nessun tuning degli iperparametri.

## Valutazione
| Modello | ROC-AUC | PR-AUC | Precisione a recall 90% |
| --- | ---: | ---: | ---: |
| Random forest (4 feature del catalogo) | 0.936 ± 0.006 | 0.859 ± 0.018 | 0.776 ± 0.017 |
| CNN solo curve | 0.970 ± 0.006 | 0.948 ± 0.011 | 0.888 ± 0.020 |
| CNN completa (curve + feature del catalogo) | 0.973 ± 0.007 | 0.949 ± 0.019 | 0.887 ± 0.020 |
| CNN solo vista locale | 0.963 ± 0.010 | 0.941 ± 0.014 | 0.844 ± 0.038 |
| CNN solo vista globale | 0.958 ± 0.006 | 0.928 ± 0.014 | 0.813 ± 0.032 |

*Media ± deviazione standard su 5 divisioni (test di 685–745 esempi). Differenza di PR-AUC CNN solo curve − random
forest: +0.089 ± 0.020, migliore in 5 su 5 divisioni. Le feature del catalogo non migliorano in modo misurabile.*

Controlli: con etichette permutate tra stelle la ROC-AUC è 0.487 ± 0.020 (CNN) e 0.494 ± 0.019 (random forest), cioè ~0.5, quindi
nessun leakage evidente nello split e nelle viste. Il solo SNR come punteggio dà ROC-AUC 0.491 (intervallo 0.447–0.534).

## Prestazioni per regime
Test dei 5 seed messi insieme (3563 righe, 2973 KOI distinti; la stessa stella può comparire in più
divisioni, quindi le righe non sono indipendenti), alla soglia scelta sulla validazione.

| Periodo (giorni) | Esempi | Positivi | Recall | Precisione |
| --- | ---: | ---: | ---: | ---: |
| fino a 1.5 | 606 | 34 | 0.47 | 0.53 |
| 1.5–3 | 481 | 105 | 0.76 | 0.79 |
| 3–10 | 844 | 436 | 0.92 | 0.90 |
| 10–30 | 774 | 454 | 0.93 | 0.92 |
| 30–100 | 439 | 222 | 0.92 | 0.94 |
| oltre 100 | 419 | 63 | 0.86 | 0.70 |

| Profondità (ppm) | Esempi | Positivi | Recall | Precisione |
| --- | ---: | ---: | ---: | ---: |
| fino a 200 | 981 | 305 | 0.81 | 0.82 |
| 200–500 | 801 | 407 | 0.90 | 0.89 |
| 500–1000 | 543 | 347 | 0.94 | 0.94 |
| 1000–2500 | 353 | 178 | 0.94 | 0.90 |
| 2500–10000 | 213 | 49 | 0.98 | 0.87 |
| oltre 10000 | 672 | 28 | 0.68 | 0.63 |

| SNR del catalogo | Esempi | Positivi | Recall | Precisione |
| --- | ---: | ---: | ---: | ---: |
| fino a 7.1 | 141 | 2 | 0.50 | 0.20 |
| 7.1–10 | 236 | 18 | 0.33 | 0.27 |
| 10–15 | 460 | 137 | 0.73 | 0.75 |
| 15–25 | 704 | 385 | 0.86 | 0.89 |
| 25–50 | 710 | 432 | 0.96 | 0.93 |
| 50–100 | 359 | 182 | 0.97 | 0.93 |
| 100–500 | 469 | 134 | 0.96 | 0.94 |
| oltre 500 | 484 | 24 | 0.79 | 0.68 |

## Calibrazione e soglia
Il punteggio grezzo non è una probabilità: errore di calibrazione atteso 0.055 ± 0.010 (Brier 0.069 ± 0.007). Con una calibrazione isotonica
adattata sulla validazione: 0.024 ± 0.007 (Brier 0.062 ± 0.008); la calibrazione isotonica produce valori a gradini (anche 0 e 1 esatti),
quindi non va letta come certezza. Soglia per un recall obiettivo di 0.90 scelta sulla validazione: sul test recall
0.887 ± 0.022 e precisione 0.892 ± 0.019; per l'obiettivo più alto recall 0.944 ± 0.009 e precisione
0.812 ± 0.036.

## Completezza con transiti iniettati
Transiti sintetici (profilo trapezoidale, durata ∝ P^(1/3)) in curve Kepler reali tranquille delle stelle di test, transiti dei KOI
veri eliminati; 2506 iniezioni valide su 2520. Quota riconosciuta alla soglia di validazione:

| Profondità \ periodo (giorni) | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 ppm | 0.93 | 0.88 | 0.88 | 0.80 | 0.71 | 0.67 | 0.41 |
| 500 ppm | 0.98 | 0.97 | 0.95 | 0.88 | 0.85 | 0.87 | 0.83 |
| 1000 ppm | 1.00 | 0.98 | 0.98 | 0.98 | 0.92 | 0.93 | 0.85 |
| 2000 ppm | 1.00 | 1.00 | 1.00 | 0.97 | 0.98 | 0.93 | 0.92 |
| 5000 ppm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.97 |
| 10000 ppm | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |

Con effemeride nota si misura il classificatore, non la ricerca del periodo; le iniezioni sono solo pianeti (completezza, non tasso di
falsi positivi).

## Robustezza
| Perturbazione | KOI tenuti | PR-AUC | Recall | Precisione |
| --- | ---: | ---: | ---: | ---: |
| Originale | 713/713 | 0.948 ± 0.011 | 0.897 ± 0.012 | 0.883 |
| Primi 730 d | 664/713 | 0.931 ± 0.019 | 0.793 ± 0.016 | 0.904 |
| Primi 365 d | 636/713 | 0.898 ± 0.020 | 0.643 ± 0.036 | 0.912 |
| Rumore x2 | 713/713 | 0.883 ± 0.027 | 0.623 ± 0.038 | 0.897 |
| Rumore x4 | 713/713 | 0.741 ± 0.057 | 0.336 ± 0.022 | 0.870 |

Rumore gaussiano bianco e troncamento della baseline: non coprono il rumore correlato (rotazione stellare, sistematiche).

## Trasferimento a TESS
| Approccio | ROC-AUC | PR-AUC | Precisione a recall 90% |
| --- | ---: | ---: | ---: |
| Modello Kepler così com'è | 0.837 ± 0.014 | 0.840 ± 0.035 | 0.762 ± 0.018 |
| Modello Kepler affinato su TESS | 0.868 ± 0.035 | 0.866 ± 0.057 | 0.780 ± 0.028 |
| CNN addestrata solo su TESS | 0.826 ± 0.034 | 0.855 ± 0.038 | 0.711 ± 0.037 |
| Random forest (3 feature) | 0.725 ± 0.027 | 0.742 ± 0.043 | 0.675 ± 0.012 |

Test di 148–158 esempi per divisione. Il modello Kepler così com'è su tutti i 1543 candidati TESS: ROC-AUC
0.827–0.848 a seconda del seed. Affinamento − modello così com'è: PR-AUC +0.026 ± 0.033 (4 su 5
divisioni); affinamento − da zero: +0.012 ± 0.063. Controllo con etichette permutate: ROC-AUC 0.566 ± 0.064.

## Limiti, rischi e usi impropri
- **Tetto strutturale**: solo curva di luce; niente centroide, spettri o immagini. Molti falsi positivi hanno transiti puliti e
  indistinguibili da quelli di un pianeta.
- **Debolezze misurate**: periodi brevi, dip molto profondi (binarie a eclisse), SNR basso, curve corte o rumorose; con meno dati la soglia
  fissa fa perdere recall.
- **Etichette imperfette** e classi dipendenti da periodo e profondità: possibili scorciatoie; i controlli fatti (ablation, etichette
  permutate, iniezione) non le escludono tutte.
- **Statistica**: test di poche centinaia di esempi, 5 seed, una sola famiglia di modelli; differenze piccole non sono significative.
  La GPU non è deterministica: due esecuzioni della stessa configurazione danno medie leggermente diverse.
- **Rischio di uso improprio**: presentare il punteggio come una scoperta o una probabilità calibrata. Il progetto lo dichiara in ogni
  schermata; chi lo riusa deve fare lo stesso.
- **Testi generati**: le schede in italiano sono riformulate da un modello locale (`gemma3:12b`) a partire da numeri già calcolati e
  passano controlli automatici (segnalate: 98% con i dati come JSON, 0% con righe etichettate, su 50
  candidati); i controlli sono euristiche e i difetti più gravi (numeri veri nel ruolo sbagliato) li ha trovati la lettura a mano.

## Riproducibilità
Seed espliciti, configurazione e metriche in JSON, versioni delle librerie nei report; split per stella con `StratifiedGroupKFold`.
Gli snapshot dei report in `docs/results/` alimentano questo documento (`python scripts/render_docs.py`). Passi completi nel README.

## Riferimenti
Vedi `docs/SCIENCE_NOTES.md` (titoli, anni e DOI verificati su Crossref): Shallue & Vanderburg (2018), Kovács et al. (2002),
Thompson et al. (2018), Ansdell et al. (2018), Borucki et al. (2010), Ricker et al. (2015), Akeson et al. (2013).
