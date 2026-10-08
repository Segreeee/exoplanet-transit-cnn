# Note scientifiche

## Il problema in breve
Un pianeta che passa davanti alla sua stella ne riduce il flusso per qualche ora, ripetutamente
(transito). Kepler ha osservato ~150.000 stelle per 4 anni. Un algoritmo (qui BLS) trova segnali
periodici; molti pero' non sono pianeti ma **falsi positivi**: binarie a eclisse, stelle di fondo,
variabilita' stellare, sistematiche strumentali. La classificazione "pianeta vs falso positivo"
(*vetting*) e' il compito del modello.

## Perche' due viste
- **Vista globale** (2001 bin): tutta la fase orbitale. Mostra eclisse secondaria (tipica di una binaria),
  variabilita' fuori transito.
- **Vista locale** (201 bin): zoom di ±2 durate attorno al transito. Mostra la forma (a U per un pianeta,
  piu' a V per molte binarie "radenti") e la profondita' relativa.
- **Feature scalari** (4): log10 profondita', log10 periodo, durata/periodo, log1p SNR.

## Limiti fisici da dichiarare nel portfolio
- Dalla sola curva di luce non si vede la posizione del centroide (stelle di fondo), ne' lo spettro,
  ne' le immagini ad alta risoluzione: esiste un tetto alle prestazioni.
- Le etichette KOI non sono verita' di laboratorio: "CONFIRMED" e' validato statisticamente o con
  velocita' radiali, "FALSE POSITIVE" e' una disposizione del team. Contengono errori.
- Classi sbilanciate e dipendenti dal periodo e dalla profondita': un modello puo' imparare scorciatoie
  (es. "SNR alto = pianeta") invece della fisica. Controlla con ablation e con injection-recovery.
- Kepler e' il set storico: i risultati non si trasferiscono automaticamente a TESS (cadenza, durata
  delle osservazioni, strumentazione diverse).

## Cose da NON fare
- Split casuale per riga (leakage tra candidati della stessa stella).
- Rifare il detrending senza maschera sui transiti (cancella il segnale che cerchi).
- Scegliere il seed migliore e riportare quello.
- Tarare soglia o iperparametri sul test set.
- Scrivere "ho scoperto un pianeta".

## Riferimenti (VERIFICA titoli, anno e DOI prima di citarli nel README)
- Shallue & Vanderburg (2018), "Identifying Exoplanets with Deep Learning", AJ: l'idea delle due viste.
  Questo progetto e' una re-implementazione piu' piccola ispirata a quel lavoro, non una riproduzione.
- Kovacs, Zucker & Mazeh (2002), Box Least Squares: ricerca di transiti periodici.
- Thompson et al. (2018), catalogo KOI DR25 (Kepler).
- Ansdell et al. (2018): aggiunta di informazione sul centroide al modello.
- Documentazione: NASA Exoplanet Archive (tabella `cumulative` dei KOI), lightkurve, astropy (`BoxLeastSquares`).
