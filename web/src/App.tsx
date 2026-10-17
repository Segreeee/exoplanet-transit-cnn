import { useEffect, useMemo, useState } from "react";
import { LineChart } from "./LineChart";
import { ResultsSection } from "./Results";
import { f3, softProb } from "./format";
import type { Demo, Example, Outcome } from "./types";

const OUTCOME_TEXT: Record<Outcome, string> = {
  TP: "Corretto: candidato riconosciuto",
  TN: "Corretto: falso positivo scartato",
  FP: "Errore: falso positivo promosso a candidato",
  FN: "Errore: pianeta confermato scartato",
};
const OUTCOME_SHORT: Record<Outcome, string> = { TP: "giusto", TN: "giusto", FP: "errore", FN: "errore" };
const isError = (o: Outcome) => o === "FP" || o === "FN";

export function App() {
  const [demo, setDemo] = useState<Demo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/demo.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setDemo)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <>
      <Hero demo={demo} />
      <main>
        {error && (
          <p className="banner error" role="alert">
            Impossibile caricare i dati della demo ({error}). Ricarica la pagina; se persiste, il file data/demo.json non è stato pubblicato.
          </p>
        )}
        {!demo && !error && <p className="muted">Caricamento dei dati…</p>}
        <Pipeline />
        {demo && (
          <>
            <Explorer demo={demo} />
            <ResultsSection demo={demo} />
            <Limits demo={demo} />
          </>
        )}
      </main>
      <footer>
        <p>
          Progetto personale di un ingegnere informatico. Ogni numero di questa pagina viene da file generati dal repository (
          <code>demo.json</code>), non è scritto nel codice dell&apos;interfaccia.
        </p>
      </footer>
    </>
  );
}

function Hero({ demo }: { demo: Demo | null }) {
  // esempio in evidenza: il candidato riconosciuto con SNR più alto (scelta dichiarata, non a mano)
  const featured = useMemo(() => {
    if (!demo) return null;
    const tp = demo.examples.filter((e) => e.outcome === "TP");
    return tp.length ? tp.reduce((a, b) => (b.snr > a.snr ? b : a)) : null;
  }, [demo]);
  return (
    <header className="hero">
      <div className="hero-text">
        <h1>Pianeta o falso allarme?</h1>
        <p className="lead">
          Quando un pianeta passa davanti alla sua stella, la luce cala di pochissimo e a intervalli regolari. Una rete neurale impara a
          distinguere questi cali dai tanti impostori, come le coppie di stelle che si eclissano. Qui sotto la puoi sfidare su curve di luce
          vere del telescopio Kepler, errori compresi.
        </p>
        {demo && (
          <p className="honest" role="note">
            Dati reali: {demo.results.kepler_dataset.n_examples} candidati della tabella KOI della NASA. Il risultato è il punteggio di un
            classificatore, non una scoperta: dalla sola curva di luce non si vedono la posizione sul cielo né lo spettro, e per questo esiste
            un limite alle prestazioni.
          </p>
        )}
      </div>
      {featured && (
        <figure className="hero-plot">
          <LineChart
            x={featured.local_view.map((_, i) => -2 + (4 * i) / (featured.local_view.length - 1))}
            y={featured.local_view}
            xLabel="durate dal centro del transito"
            yLabel="flusso normalizzato"
            height={260}
            caption={`Transito di ${featured.id}`}
          />
          <figcaption>
            {featured.id}: periodo {featured.period_days} giorni, profondità {featured.depth_ppm} ppm. Zoom sul transito, curva ripiegata su
            tutti i passaggi osservati da Kepler.
          </figcaption>
        </figure>
      )}
    </header>
  );
}

function Pipeline() {
  return (
    <section>
      <h2>Come funziona</h2>
      <ol className="steps">
        <li>
          <b>Curva di luce</b>
          <span>Il flusso della stella nel tempo, corretto per le sistematiche dello strumento.</span>
        </li>
        <li>
          <b>Appiattimento</b>
          <span>Si toglie la variabilità lenta della stella, proteggendo i transiti già noti dal catalogo.</span>
        </li>
        <li>
          <b>Vista globale e locale</b>
          <span>La curva ripiegata sul periodo: intera, per vedere eclissi secondarie, e con zoom sul transito, per vederne la forma.</span>
        </li>
        <li>
          <b>Rete a due torri</b>
          <span>Due reti 1D leggono le due viste e producono un punteggio. Non una conferma.</span>
        </li>
      </ol>
    </section>
  );
}

function Explorer({ demo }: { demo: Demo }) {
  const [selected, setSelected] = useState(0);
  const [guess, setGuess] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [onlyErrors, setOnlyErrors] = useState(false);

  const visible = useMemo(
    () => demo.examples.map((e, i) => ({ e, i })).filter(({ e }) => !(onlyErrors && !isError(e.outcome))),
    [demo, onlyErrors],
  );
  const choose = (i: number) => {
    setSelected(i);
    setRevealed(false);
  };
  const m = demo.meta;
  const showAnswer = !guess || revealed;

  return (
    <section id="esplora">
      <h2>Guarda una curva e decidi</h2>
      <p className="muted">
        {m.n_examples_shown} candidati scelti tra quelli che il modello non ha mai visto in addestramento (split per stella): per ogni esito
        (giusto o sbagliato) ho preso un numero fisso di esempi a caso, quindi gli errori compaiono molto più spesso che nella realtà. Le
        metriche della sezione successiva usano invece tutto il test ({m.test_set_size} candidati per divisione).
      </p>
      <div className="controls">
        <label className="check">
          <input
            type="checkbox"
            checked={guess}
            onChange={(ev) => {
              setGuess(ev.target.checked);
              setRevealed(false);
            }}
          />
          Nascondi la risposta finché non la chiedo
        </label>
        <label className="check">
          <input type="checkbox" checked={onlyErrors} onChange={(ev) => setOnlyErrors(ev.target.checked)} />
          Mostra solo gli errori
        </label>
      </div>
      <div className="explorer">
        <ul className="list" aria-label="Candidati">
          {visible.map(({ e, i }) => (
            <li key={e.id}>
              <button className={i === selected ? "item active" : "item"} onClick={() => choose(i)} aria-current={i === selected}>
                <span>{e.id}</span>
                {showAnswer && <span className={`tag ${isError(e.outcome) ? "bad" : "good"}`}>{OUTCOME_SHORT[e.outcome]}</span>}
              </button>
            </li>
          ))}
          {visible.length === 0 && <li className="muted">Nessun errore da mostrare.</li>}
        </ul>
        <ExampleView
          e={demo.examples[selected]}
          threshold={m.threshold_score}
          showAnswer={showAnswer}
          canReveal={guess && !revealed}
          onReveal={() => setRevealed(true)}
          reportModel={m.report_model}
        />
      </div>
    </section>
  );
}

function ExampleView(props: {
  e: Example;
  threshold: number;
  showAnswer: boolean;
  canReveal: boolean;
  onReveal: () => void;
  reportModel: string | null;
}) {
  const { e, threshold, showAnswer, canReveal, onReveal, reportModel } = props;
  const gx = e.global_view.map((_, i) => -0.5 + i / (e.global_view.length - 1));
  const lx = e.local_view.map((_, i) => -2 + (4 * i) / (e.local_view.length - 1));
  const span = e.light_curve.t.length ? e.light_curve.t[e.light_curve.t.length - 1] - e.light_curve.t[0] : 0;
  return (
    <article className="detail">
      <header className="detail-head">
        <h3>{e.id}</h3>
        <p className="muted">Stella KIC {e.kepid}</p>
      </header>

      <dl className="facts">
        <div>
          <dt>Periodo</dt>
          <dd>{e.period_days} giorni</dd>
        </div>
        <div>
          <dt>Durata del transito</dt>
          <dd>{e.duration_hours} ore</dd>
        </div>
        <div>
          <dt>Profondità</dt>
          <dd>{e.depth_ppm} ppm</dd>
        </div>
        <div>
          <dt>SNR (catalogo)</dt>
          <dd>{e.snr}</dd>
        </div>
      </dl>

      <h4>Curva di luce</h4>
      <LineChart
        x={e.light_curve.t}
        y={e.light_curve.f}
        xLabel="tempo (giorni)"
        yLabel="flusso"
        dots
        marks={e.transits}
        caption="Curva di luce con i transiti previsti"
      />
      <p className="caption">
        Finestra di {Math.round(span)} giorni. Le linee verticali sono i transiti previsti dal catalogo: guarda se il flusso cala proprio lì, e
        quanto è rumoroso rispetto al calo.
      </p>

      <div className="views">
        <div>
          <h4>Vista globale</h4>
          <LineChart x={gx} y={e.global_view} xLabel="fase orbitale" yLabel="flusso normalizzato" height={200} width={420} />
        </div>
        <div>
          <h4>Vista locale</h4>
          <LineChart x={lx} y={e.local_view} xLabel="durate dal centro" yLabel="flusso normalizzato" height={200} width={420} />
        </div>
      </div>

      <div className="verdict">
        {!showAnswer ? (
          <div className="guess">
            <p>Cosa ne pensi: pianeta o impostore? Poi confronta con la risposta del classificatore e con l&apos;archivio della NASA.</p>
            {canReveal && (
              <button className="primary" onClick={onReveal}>
                Mostra la risposta
              </button>
            )}
          </div>
        ) : (
          <Verdict e={e} threshold={threshold} />
        )}
      </div>

      {showAnswer && (
        <div className="report">
          <p>{e.report}</p>
          <p className="muted small">
            {e.report_source === "template"
              ? "Testo prodotto da un modello fisso a partire dai numeri sopra."
              : `Testo riformulato dal modello locale ${reportModel ?? e.report_source} a partire dai numeri sopra e controllato automaticamente: può contenere errori. Nel testo la probabilità è quella calibrata e la soglia è la sua corrispondente; nel riquadro sopra il punteggio è quello grezzo della rete.`}
          </p>
        </div>
      )}
    </article>
  );
}

function Verdict({ e, threshold }: { e: Example; threshold: number }) {
  const bad = isError(e.outcome);
  return (
    <div className="verdict-body">
      <p className={`outcome ${bad ? "bad" : "good"}`}>{OUTCOME_TEXT[e.outcome]}</p>
      <div className="bar" aria-hidden>
        <div className="fill" style={{ width: `${Math.min(e.score, 1) * 100}%` }} />
        <div className="thr" style={{ left: `${Math.min(threshold, 1) * 100}%` }} />
      </div>
      <p>
        Punteggio del classificatore <b>{f3(e.score)}</b>, soglia {f3(threshold)} scelta sulla validazione: <b>{e.predicted_candidate ? "candidato" : "scartato"}</b>.
        Probabilità calibrata: {softProb(e.prob_calibrated)}.
      </p>
      <p>
        Archivio della NASA (KOI): <b>{e.label === 1 ? "pianeta confermato" : "falso positivo"}</b>.
      </p>
    </div>
  );
}

function Limits({ demo }: { demo: Demo }) {
  const r = demo.results;
  return (
    <section>
      <h2>Limiti</h2>
      <ul className="limits">
        <li>
          Un punteggio alto non è una scoperta: è un candidato da verificare con altri strumenti. Le etichette dell&apos;archivio sono
          decisioni di un gruppo di ricerca e possono contenere errori.
        </li>
        <li>
          Dalla sola curva di luce non si vedono la posizione del segnale sul cielo, gli spettri né le immagini ad alta risoluzione: molti falsi
          positivi hanno un transito pulito e identico a quello di un pianeta, e il modello non può distinguerli.
        </li>
        <li>
          I risultati vengono da {r.cnn.n_seeds} divisioni per stella con test di poche centinaia di esempi, iperparametri non ottimizzati e
          un solo dataset ({r.kepler_dataset.n_examples} candidati): differenze piccole tra modelli non sono significative.
        </li>
        <li>
          Il modello sbaglia dove il segnale è debole, i periodi sono brevi o le curve sono corte; i numeri per fascia sono sopra. Le stesse
          fasce hanno pochi positivi: le stime sono incerte.
        </li>
        {r.tess && (
          <li>
            Su TESS ho usato solo le stelle scaricate prima di finire lo spazio su disco ({r.tess.dataset.n_stars} stelle): non è l&apos;intero
            catalogo e i test sono piccoli.
          </li>
        )}
        <li>
          Il punteggio grezzo non è una probabilità calibrata, e la calibrazione isotonica dà valori a gradini (anche 0 e 1 esatti): non
          vanno letti come certezze.
        </li>
        <li>
          Il testo delle schede è riformulato da un modello locale. I controlli automatici non bastano: l&apos;ho verificato leggendo un campione,
          non tutte le schede della pagina.
        </li>
      </ul>
    </section>
  );
}
