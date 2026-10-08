import { useEffect, useMemo, useState } from "react";
import { LineChart } from "./LineChart";
import type { Demo, Example, Metrics } from "./types";

const KIND_LABEL: Record<Example["kind"], string> = {
  planet: "Pianeta (segnale iniettato)",
  eb: "Binaria a eclisse (falso positivo)",
  noise: "Solo rumore (falso positivo)",
};

const f3 = (v: number | null) => (v === null || Number.isNaN(v) ? "n/d" : v.toFixed(3));

function outcome(e: Example): "TP" | "FP" | "FN" | "TN" {
  if (e.label === 1) return e.predicted_candidate ? "TP" : "FN";
  return e.predicted_candidate ? "FP" : "TN";
}

const OUTCOME_TEXT = {
  TP: "Corretto: pianeta riconosciuto",
  TN: "Corretto: falso positivo scartato",
  FP: "Errore: falso positivo promosso a candidato",
  FN: "Errore: pianeta scartato",
};

export function App() {
  const [demo, setDemo] = useState<Demo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);
  const [onlyErrors, setOnlyErrors] = useState(false);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/demo.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setDemo)
      .catch((e: Error) => setError(e.message));
  }, []);

  const visible = useMemo(
    () => (demo ? demo.examples.map((e, i) => ({ e, i })).filter(({ e }) => !onlyErrors || outcome(e)[0] === "F") : []),
    [demo, onlyErrors],
  );

  return (
    <>
      <header className="hero">
        <p className="eyebrow">Progetto personale · ML per la fisica</p>
        <h1>Esopianeti dalle curve di luce</h1>
        <p className="lead">
          Una CNN a due viste, ispirata ad AstroNet, che distingue candidati pianeta da falsi positivi a partire dal calo
          di luminosità di una stella. Pipeline completa: rilevamento dei transiti (BLS), viste ripiegate, rete neurale,
          valutazione con split per stella.
        </p>
        <div className="banner" role="note">
          <strong>Stato onesto del lavoro.</strong> Tutto ciò che vedi qui è calcolato su <strong>dati sintetici</strong>:
          dimostra che la pipeline funziona, non che il modello riconosca esopianeti veri. La validazione su dati Kepler
          reali non è ancora stata fatta.
        </div>
      </header>

      <main>
        <section>
          <h2>La pipeline</h2>
          <ol className="steps">
            <li>
              <b>Curva di luce</b>
              <span>Flusso della stella nel tempo, con variabilità stellare e rumore.</span>
            </li>
            <li>
              <b>Detrending + BLS</b>
              <span>Si rimuove la variabilità lenta e si cerca un calo periodico (Box Least Squares).</span>
            </li>
            <li>
              <b>Vista globale e locale</b>
              <span>La curva ripiegata sul periodo, intera e con zoom sul transito.</span>
            </li>
            <li>
              <b>CNN a due torri</b>
              <span>Due reti 1D, più 4 feature scalari: restituisce un punteggio, non una conferma.</span>
            </li>
          </ol>
        </section>

        {error && <p className="banner error">Impossibile caricare i dati della demo ({error}).</p>}
        {!demo && !error && <p className="muted">Caricamento…</p>}

        {demo && (
          <>
            <section>
              <h2>Esplora gli esempi del test set</h2>
              <p className="muted">
                {demo.examples.length} esempi mai visti in addestramento (split per stella), con la risposta del modello.
                Includo gli errori: guardali.
              </p>
              <label className="check">
                <input type="checkbox" checked={onlyErrors} onChange={(ev) => setOnlyErrors(ev.target.checked)} /> Mostra
                solo gli errori
              </label>
              <div className="explorer">
                <ul className="list" aria-label="Esempi">
                  {visible.map(({ e, i }) => (
                    <li key={e.id + i}>
                      <button className={i === selected ? "item active" : "item"} onClick={() => setSelected(i)}>
                        <span>{e.id}</span>
                        <span className={`chip ${outcome(e)}`}>{outcome(e)}</span>
                      </button>
                    </li>
                  ))}
                  {visible.length === 0 && <li className="muted">Nessun errore da mostrare.</li>}
                </ul>
                <ExampleView e={demo.examples[selected]} threshold={demo.meta.threshold_cnn} />
              </div>
            </section>

            <ResultsSection demo={demo} />
          </>
        )}

        <section>
          <h2>Limiti</h2>
          <ul className="limits">
            <li>Dati sintetici: la difficoltà reale (variabilità stellare vera, sistematiche, binarie complesse) è molto maggiore.</li>
            <li>Test set minuscolo: differenze tra modelli di pochi punti percentuali non sono statisticamente significative.</li>
            <li>Con la sola curva di luce non si vedono centroide, spettri o immagini ad alta risoluzione: esiste un tetto alle prestazioni.</li>
            <li>Un punteggio alto non è una scoperta: è un candidato da verificare con altri strumenti.</li>
            <li>Lo script di download dei dati Kepler non è ancora stato verificato contro i server reali.</li>
          </ul>
        </section>
      </main>

      <footer>
        <p>
          Prossimi passi: dati Kepler reali, baseline sul solo SNR, 5 seed con deviazione standard, ablation, test di
          iniezione-recupero. Il progetto è in costruzione: i risultati qui sopra vengono generati dal repository, non
          scritti a mano.
        </p>
      </footer>
    </>
  );
}

function ExampleView({ e, threshold }: { e: Example; threshold: number }) {
  const o = outcome(e);
  const globalX = e.global_view.map((_, i) => -0.5 + i / (e.global_view.length - 1));
  const localX = e.local_view.map((_, i) => -2 + (4 * i) / (e.local_view.length - 1));
  return (
    <div className="detail">
      <div className="detail-head">
        <div>
          <h3>{e.id}</h3>
          <p className="muted">Etichetta vera: {KIND_LABEL[e.kind]}</p>
        </div>
        <span className={`chip big ${o}`}>{OUTCOME_TEXT[o]}</span>
      </div>

      <div className="score">
        <div className="bar" aria-hidden>
          <div className="fill" style={{ width: `${Math.min(e.prob_planet, 1) * 100}%` }} />
          <div className="thr" style={{ left: `${Math.min(threshold, 1) * 100}%` }} />
        </div>
        <p>
          Punteggio del classificatore <b>{e.prob_planet.toFixed(3)}</b> · soglia {threshold.toFixed(3)} (scelta sulla
          validazione) → <b>{e.predicted_candidate ? "candidato" : "scartato"}</b>
        </p>
      </div>

      <dl className="facts">
        <div><dt>Periodo</dt><dd>{e.period_days} giorni</dd></div>
        <div><dt>Durata</dt><dd>{e.duration_hours} ore</dd></div>
        <div><dt>Profondità</dt><dd>{e.depth_ppm} ppm</dd></div>
        <div><dt>SNR (BLS)</dt><dd>{e.snr}</dd></div>
      </dl>

      <h4>Curva di luce (medie su intervalli)</h4>
      <LineChart x={e.light_curve.t} y={e.light_curve.f} xLabel="tempo (giorni)" yLabel="flusso" dots />
      <h4>Vista globale (ripiegata sul periodo)</h4>
      <LineChart x={globalX} y={e.global_view} xLabel="fase orbitale" yLabel="flusso normalizzato" color="var(--blue)" />
      <h4>Vista locale (±2 durate attorno al transito)</h4>
      <LineChart x={localX} y={e.local_view} xLabel="durate dal centro" yLabel="flusso normalizzato" color="var(--green)" />

      <p className="report">{e.report}</p>
      <p className="muted small">Testo generato da un modello deterministico a partire dai numeri sopra (nessun LLM in questa pagina).</p>
    </div>
  );
}

function Row({ name, m, extra }: { name: string; m: Metrics; extra?: string }) {
  return (
    <tr>
      <td>
        {name}
        {extra && <span className="muted small"> {extra}</span>}
      </td>
      <td>{f3(m.roc_auc)}</td>
      <td>{f3(m.pr_auc)}</td>
      <td>{f3(m.precision_at_recall90)}</td>
      <td>{f3(m.precision)}</td>
      <td>{f3(m.recall)}</td>
      <td>{`${m.tp}/${m.fp}/${m.fn}/${m.tn}`}</td>
    </tr>
  );
}

function ResultsSection({ demo }: { demo: Demo }) {
  const { metrics: m, meta } = demo;
  const n = m.cnn_val_threshold.n;
  return (
    <section>
      <h2>Risultati (test set sintetico, n = {n})</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Modello</th>
              <th>ROC-AUC</th>
              <th>PR-AUC</th>
              <th>Prec@Rec90</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>TP/FP/FN/TN</th>
            </tr>
          </thead>
          <tbody>
            <Row name="CNN" m={m.cnn_val_threshold} extra={`soglia ${meta.threshold_cnn.toFixed(3)} (validazione)`} />
            <Row name="Random forest (solo feature scalari)" m={m.rf_val_threshold} extra={`soglia ${meta.threshold_rf.toFixed(3)} (validazione)`} />
            <Row name="CNN con soglia 0.5" m={m.cnn_default_threshold} extra="punteggi non calibrati" />
          </tbody>
        </table>
      </div>
      <ul className="notes">
        <li>
          Su ROC-AUC e PR-AUC la CNN è sopra la baseline, ma con {n} esempi e un solo seed non è una differenza
          dimostrata. Alla soglia scelta sulla validazione, le due classificano in modo simile.
        </li>
        <li>
          Con soglia 0.5 la CNN non riconosce alcun pianeta: il ranking è buono ma i punteggi non sono calibrati. La
          soglia va scelta sulla validazione, mai sul test.
        </li>
        <li>
          Addestramento: {meta.epochs} epoche massime, miglior epoca {meta.best_epoch}, seed {meta.seed}, device{" "}
          {meta.device}, split {Object.entries(meta.split_sizes).map(([k, v]) => `${k} ${v}`).join(" / ")}.
        </li>
      </ul>
    </section>
  );
}
