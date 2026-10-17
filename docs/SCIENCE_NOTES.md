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
  La calibrazione isotonica sulla validazione riduce l'errore di calibrazione (ECE) e la soglia per un recall obiettivo scelta
  sulla validazione si trasferisce bene al test; i valori sono nella model card (generata dai report).
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

## Riferimenti (titoli, anni, riviste e DOI verificati su Crossref; ASCL su ascl.net)
Verifica fatta con una richiesta a `api.crossref.org/works/<DOI>` per ogni voce: titolo, primi autori, anno, rivista, volume e
pagina coincidono con quelli sotto.
- Shallue, C. J. & Vanderburg, A. (2018), *Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around
  Kepler-80 and an Eighth Planet around Kepler-90*, The Astronomical Journal 155, 94. doi:10.3847/1538-3881/aa9e09.
  L'idea delle due viste; questo progetto e' una re-implementazione piu' piccola ispirata a quel lavoro, **non** una riproduzione.
- Kovacs, G., Zucker, S. & Mazeh, T. (2002), *A box-fitting algorithm in the search for periodic transits*, Astronomy &
  Astrophysics 391, 369-377. doi:10.1051/0004-6361:20020802. Box Least Squares (usato nel percorso sintetico e disponibile in
  `preprocess.run_bls`; sui dati reali le effemeridi vengono dal catalogo).
- Thompson, S. E. et al. (2018), *Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog with Measured
  Completeness and Reliability Based on Data Release 25*, The Astrophysical Journal Supplement Series 235, 38.
  doi:10.3847/1538-4365/aab4f9. Catalogo KOI DR25.
- Ansdell, M. et al. (2018), *Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning*, The
  Astrophysical Journal Letters 869, L7. doi:10.3847/2041-8213/aaf23b. Informazione sul centroide.
- Borucki, W. J. et al. (2010), *Kepler Planet-Detection Mission: Introduction and First Results*, Science 327, 977-980.
  doi:10.1126/science.1185402.
- Ricker, G. R. et al., *Transiting Exoplanet Survey Satellite*, Journal of Astronomical Telescopes, Instruments, and Systems 1(1),
  014003. doi:10.1117/1.JATIS.1.1.014003. Crossref data la pubblicazione al 24 ottobre 2014 (volume 1); spesso e' citato come 2015.
- Akeson, R. L. et al. (2013), *The NASA Exoplanet Archive: Data and Tools for Exoplanet Research*, Publications of the Astronomical
  Society of the Pacific 125, 989-999. doi:10.1086/672273. Tabelle `cumulative` (KOI) e `toi`.
- Stumpe, M. C. et al. (2012), *Kepler Presearch Data Conditioning I - Architecture and Algorithms for Error Correction in Kepler Light
  Curves*, PASP 124, 985-999. doi:10.1086/667698; Smith, J. C. et al. (2012), *Kepler Presearch Data Conditioning II - A Bayesian
  Approach to Systematic Error Correction*, PASP 124, 1000-1014. doi:10.1086/667697. Il flusso PDCSAP che usiamo.
- Zadrozny, B. & Elkan, C. (2002), *Transforming classifier scores into accurate multiclass probability estimates*, Proceedings of
  the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 694-699. doi:10.1145/775047.775151.
  Calibrazione con regressione isotonica.
- Lightkurve Collaboration (2018), *Lightkurve: Kepler and TESS time series analysis in Python*, Astrophysics Source Code Library,
  record ascl:1812.013 (autore e voce verificati su ascl.net; titolo riportato dalla pagina del software).
- Documentazione: NASA Exoplanet Archive (tabelle `cumulative` e `toi`), astropy (`BoxLeastSquares`), scikit-learn
  (`IsotonicRegression`, `StratifiedGroupKFold`).
