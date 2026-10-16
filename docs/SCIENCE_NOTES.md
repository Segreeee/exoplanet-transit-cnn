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

## Dataset Kepler: scelte di costruzione
- **Piu' KOI per stella**: ogni KOI e' un esempio, ma nella curva ci sono anche i transiti degli altri KOI
  (anche CANDIDATE e FALSE POSITIVE). Li eliminiamo prima di ripiegare. Nota di onesta': le viste usano la
  mediana per bin, che gia' sopprime un segnale estraneo con periodo non commensurabile; la maschera conta
  soprattutto per periodi vicini a rapporti interi (es. 2:1), dove i transiti dell'altro segnale cadono sempre
  alle stesse fasi (vedi `test_exclude_removes_other_planets_transits_from_folded_view`).
- **Buchi**: si scarta il KOI se i transiti realmente osservati (>= 3 cadenze in transito) sono meno di 3.
  La copertura media non basta: una stella al 90% puo' non avere nessun transito osservato.
- **SNR**: nessun taglio in costruzione. Un SNR mancante (NaN nella tabella KOI) e' uno scarto, non uno 0.
  Le metriche vanno riportate anche per fascia di SNR: "SNR alto = pianeta" e' una scorciatoia nota.

## Valutazione robusta: cosa significano i numeri (Fase 4)
- **Calibrazione**: il punteggio della CNN non e' una probabilita' (addestrata con `pos_weight` per lo sbilanciamento).
  La calibrazione isotonica sulla validazione riduce l'ECE da 0.055 a 0.024; la soglia per un recall obiettivo scelta
  sulla validazione si trasferisce bene al test (obiettivo 0.90 -> 0.887).
- **Injection-recovery**: si inietta un transito sintetico in una curva reale e si chiede al classificatore se lo
  riconosce. Con l'effemeride nota misura il *classificatore*, non la *rivelazione* (il BLS, che cerca il periodo, qui non
  gira): un pianeta che il BLS non troverebbe non compare come mancato. Le iniezioni sono tutte pianeti, quindi
  si misura la completezza ma non il tasso di falsi positivi.
- **Durata del transito**: per una stella tipo Sole e un transito centrale scala come P^(1/3) (a parita' di densita'
  stellare un'orbita piu' larga ha un pianeta piu' lento, ma la corda sul disco stellare cresce solo piano).
- **Rumore e baseline**: perturbazioni con rumore gaussiano bianco e troncamento della baseline non coprono il rumore
  correlato (rotazione stellare, sistematiche): sono limiti inferiori della fragilita', non una simulazione completa.
- **Pooling dei seed**: le tabelle per fascia mettono insieme i test di seed diversi; la stessa stella puo' comparire
  piu' volte, quindi le righe non sono indipendenti e gli intervalli andrebbero fatti per stella.

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
