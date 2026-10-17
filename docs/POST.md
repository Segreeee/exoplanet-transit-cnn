<!-- GENERATO da docs/templates/POST.md.tmpl con scripts/render_docs.py a partire da docs/results/*.json.
     Non modificare a mano: modifica il modello e rigenera. -->
# Pianeta o falso allarme? Una rete neurale sulle curve di luce di Kepler, con i suoi limiti

## Il problema
Quando un pianeta passa davanti alla sua stella, la luce cala di una frazione minuscola e a intervalli regolari. Kepler ha
osservato oltre centomila stelle per quattro anni, e molti cali non sono pianeti: sono coppie di stelle che si eclissano, stelle di fondo,
variabilità della stella, difetti dello strumento. Decidere quali sono veri candidati si chiama *vetting*. Io sono un ingegnere
informatico, non un astronomo: volevo capire quanto di questo lavoro può fare una rete neurale che guarda soltanto la curva di luce.

## Cosa ho costruito
Una CNN a due torri: una legge la curva ripiegata sul periodo del candidato (per vedere eclissi secondarie), l'altra uno zoom sul
transito (per vederne la forma). L'ho addestrata su 7181 candidati reali della tabella KOI della NASA
(2709 confermati, 4472 falsi positivi) e confrontata con una random forest sulle quattro feature del catalogo. Le
regole che mi sono dato: split sempre per stella, soglia scelta sulla validazione, 5 seed con media e deviazione standard,
controlli con etichette permutate e nessun numero scritto a mano nei documenti.

## Cosa ho trovato
- La CNN solo curve fa PR-AUC 0.948 ± 0.011 contro 0.859 ± 0.018 della random forest: meglio in 5 su 5 divisioni (differenza
  +0.089 ± 0.020). Le feature del catalogo non aggiungono nulla di misurabile: le curve bastano.
- Il solo SNR non serve a niente come punteggio (ROC-AUC 0.491): i segnali più forti sono spesso binarie a eclisse.
- Il modello funziona bene tra 3 e 100 giorni di periodo (recall 0.92–0.93) ma sbaglia sui periodi brevi
  (recall 0.47 su 34 positivi), sui dip molto profondi e a SNR basso. Con transiti iniettati la
  completezza scende a 0.41 nel caso peggiore (200 ppm e 128 giorni).
- Con un solo anno di dati invece di quattro il recall cala da 0.897 ± 0.012 a 0.643 ± 0.036; con rumore doppio a
  0.623 ± 0.038: la soglia, scelta su dati puliti, non segue il punteggio che si abbassa.
- Su TESS, senza vedere i suoi dati, il modello Kepler ha ROC-AUC 0.837 ± 0.014: perde, ma batte la random forest (0.725 ± 0.027).
  Affinarlo non cambia in modo distinguibile il risultato.

## Cosa non ho trovato
Un pianeta. Un punteggio alto è un candidato da verificare. Molti dei falsi positivi che il modello promuove hanno un transito pulito,
uguale a quello di un pianeta: per distinguerli servono centroide, spettri e immagini, che una curva di luce non contiene. Ho
tenuto gli errori in vista nella demo, e i test sono piccoli (poche centinaia di esempi per divisione), quindi le differenze piccole
non vanno prese per vere. Su TESS ho usato solo le stelle scaricate prima di riempire il disco.

## Una lezione sui testi generati
Per le schede in italiano ho confrontato due modelli locali. Il controllo automatico dei numeri diceva 0 problemi per entrambi; leggendo
i testi ho trovato numeri veri nel ruolo sbagliato (l'SNR "che supera la soglia", la durata scambiata col periodo). Dopo aver
etichettato meglio i dati nel prompt e aggiunto controlli per quei difetti, i testi di `gemma3:12b` risultano segnalati nel
0% dei casi (prima 98%), ma resta vero che i controlli sono euristiche e la lettura a mano non si sostituisce.

## Cosa farei dopo
Aggiungere informazione che la curva non ha (centroide), scaricare tutto TESS con più disco e ripetere con test più grandi, provare
architetture più ricche con iperparametri ottimizzati sulla validazione, iniettare anche falsi positivi per misurare il tasso di
errore e non solo la completezza, e usare rumore correlato per testare la robustezza.

Codice, demo e model card: nel repository (`README.md`, `docs/MODEL_CARD.md`).
