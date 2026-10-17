import { binLabel, f2, f3, pct, pm, signed } from "./format";
import type { BinRow, Demo, MetricName, Results } from "./types";

const METRICS: { key: MetricName; label: string }[] = [
  { key: "roc_auc", label: "ROC-AUC" },
  { key: "pr_auc", label: "PR-AUC" },
  { key: "precision_at_recall90", label: "Precisione a recall 90%" },
];

const VARIANT_LABEL: Record<string, string> = {
  full: "CNN completa (curve e feature del catalogo)",
  no_scalar: "CNN solo curve",
  local_only: "CNN solo vista locale",
  global_only: "CNN solo vista globale",
};

const TESS_LABEL: Record<string, string> = {
  kepler_as_is: "Modello Kepler così com'è",
  fine_tuned: "Modello Kepler affinato su TESS",
  tess_from_scratch: "CNN addestrata solo su TESS",
  random_forest_3_features: "Random forest (3 feature)",
};

const ROBUST_ORDER = ["originale", "primi 730 d", "primi 365 d", "rumore x2", "rumore x4"];
const cap = (k: string) => k.charAt(0).toUpperCase() + k.slice(1);

function Figure({ caption, children, wide }: { caption: React.ReactNode; children: React.ReactNode; wide?: boolean }) {
  return (
    <figure className={wide ? "fig wide" : "fig"}>
      <figcaption>{caption}</figcaption>
      {children}
    </figure>
  );
}

function ComparisonTable({ r }: { r: Results }) {
  const order = ["no_scalar", "full", "local_only", "global_only"].filter((k) => r.cnn.summary[k]);
  const ref = r.cnn.summary["no_scalar"]?.paired_diff_vs_rf.pr_auc;
  const snr = r.baselines.models["snr_only"]?.test;
  return (
    <>
      <Figure
        caption={
          <>
            Confronto con la baseline: media ± deviazione standard su {r.cnn.n_seeds} divisioni diverse per stella
            (test di {Math.min(...r.cnn.test_sizes)}–{Math.max(...r.cnn.test_sizes)} esempi ciascuna).
          </>
        }
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modello</th>
                {METRICS.map((m) => (
                  <th key={m.key}>{m.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Random forest (4 feature del catalogo)</td>
                {METRICS.map((m) => (
                  <td key={m.key}>{pm(r.cnn.baseline_random_forest[m.key])}</td>
                ))}
              </tr>
              {order.map((k) => (
                <tr key={k} className={k === "no_scalar" ? "strong" : undefined}>
                  <td>{VARIANT_LABEL[k] ?? k}</td>
                  {METRICS.map((m) => (
                    <td key={m.key}>{pm(r.cnn.summary[k].metrics[m.key])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>
      <ul className="notes">
        {ref && (
          <li>
            La CNN solo curve supera la random forest in {ref.n_seeds_better} divisioni su {r.cnn.n_seeds}: differenza di PR-AUC{" "}
            {signed(ref.mean)} ± {f3(ref.std)}.
          </li>
        )}
        <li>Aggiungere le feature del catalogo (profondità, periodo, SNR) alla CNN non cambia in modo misurabile i risultati: le curve bastano.</li>
        {snr && (
          <li>
            Il solo SNR, usato come punteggio, non separa le classi (ROC-AUC {f3(snr.roc_auc)}, intervallo {f3(snr.ci95.roc_auc[0])}–
            {f3(snr.ci95.roc_auc[1])}): i pianeti non sono i segnali più forti, perché le binarie a eclisse hanno SNR enormi.
          </li>
        )}
      </ul>
    </>
  );
}

function RegimeTable({ title, rows }: { title: string; rows: BinRow[] }) {
  return (
    <Figure caption={title}>
      <div className="table-wrap">
        <table className="regime">
          <thead>
            <tr>
              <th>Intervallo</th>
              <th>Esempi</th>
              <th>Positivi</th>
              <th>Recall</th>
              <th>Precisione</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((b, i) => (
              <tr key={b.bin}>
                <td>{binLabel(b.bin, i, rows.length - 1)}</td>
                <td>{b.n}</td>
                <td>{b.n_pos}</td>
                <td>
                  <span className="inline-bar" style={{ ["--w" as string]: `${(b.recall ?? 0) * 100}%` }} />
                  {f2(b.recall)}
                </td>
                <td>{f2(b.precision)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}

function Regimes({ r }: { r: Results }) {
  const t = r.regimes.tables;
  return (
    <>
      <p className="muted">
        Test dei {r.cnn.n_seeds} seed messi insieme ({r.regimes.n_rows} righe, {r.regimes.n_unique_kois} KOI distinti): la stessa stella
        può comparire in più seed, quindi le righe non sono indipendenti. Soglia di ogni seed scelta sulla validazione.
      </p>
      <div className="trio">
        {t["periodo_giorni"] && <RegimeTable title="Per periodo orbitale (giorni)" rows={t["periodo_giorni"]} />}
        {t["profondita_ppm"] && <RegimeTable title="Per profondità del transito (ppm)" rows={t["profondita_ppm"]} />}
        {t["snr_catalogo"] && <RegimeTable title="Per SNR del catalogo" rows={t["snr_catalogo"]} />}
      </div>
    </>
  );
}

function InjectionGrid({ r }: { r: Results }) {
  const { depths_ppm, periods, cells } = r.injection;
  const at = (d: number, p: number) => cells.find((c) => c.depth_ppm === d && c.period === p);
  return (
    <Figure
      wide
      caption={
        <>
          Completezza con transiti iniettati in curve reali tranquille: quota riconosciuta dal classificatore, per profondità e periodo
          ({r.injection.n_valid} iniezioni valide su {r.injection.n_injections}).
        </>
      }
    >
      <div className="grid-wrap">
        <table className="heat" aria-label="Completezza per profondità e periodo">
          <thead>
            <tr>
              <th>ppm \ giorni</th>
              {periods.map((p) => (
                <th key={p}>{p}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {depths_ppm.map((d) => (
              <tr key={d}>
                <th>{d}</th>
                {periods.map((p) => {
                  const c = at(d, p);
                  const v = c?.mean ?? null;
                  return (
                    <td
                      key={p}
                      className={v !== null && v > 0.55 ? "cell hi" : "cell"}
                      style={{ ["--v" as string]: v === null ? 0 : v }}
                      title={c ? `${c.n} iniezioni` : undefined}
                    >
                      {v === null ? "n/d" : f2(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}

function Robustness({ r }: { r: Results }) {
  const keys = ROBUST_ORDER.filter((k) => r.robustness[k]);
  return (
    <Figure caption="Meno dati o più rumore: soglia fissa scelta su dati puliti, media ± deviazione standard sui seed. Il confronto con le curve originali è sullo stesso sottoinsieme di KOI.">
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Perturbazione</th>
              <th>KOI tenuti</th>
              <th>PR-AUC</th>
              <th>Recall</th>
              <th>Precisione</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => {
              const e = r.robustness[k];
              return (
                <tr key={k}>
                  <td>{cap(k)}</td>
                  <td>
                    {Math.round(e.n_kept_mean)}/{Math.round(e.n_test_mean)}
                  </td>
                  <td>
                    {pm(e.pr_auc)} <span className="delta">({signed(e.pr_auc_vs_original.mean)})</span>
                  </td>
                  <td>
                    {pm(e.recall)} <span className="delta">({signed(e.recall_vs_original.mean)})</span>
                  </td>
                  <td>{f3(e.precision.mean)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}

function Calibration({ r }: { r: Results }) {
  const c = r.regimes.calibration;
  // gli obiettivi di recall sono nelle chiavi dei dati (target0.9_test_recall, ...)
  const targets = Object.keys(c)
    .map((k) => k.match(/^target([\d.]+)_test_recall$/)?.[1])
    .filter((t): t is string => Boolean(t));
  return (
    <ul className="notes">
      <li>
        Il punteggio grezzo della rete non è una probabilità: errore di calibrazione {pm(c["ece_raw"])}; dopo una calibrazione isotonica
        adattata sulla validazione scende a {pm(c["ece_isotonic"])}.
      </li>
      {targets.map((t) => (
        <li key={t}>
          Scegliendo sulla validazione la soglia per un recall di {t}, sul test si ottiene recall {pm(c[`target${t}_test_recall`])} e
          precisione {pm(c[`target${t}_test_precision`])}.
        </li>
      ))}
    </ul>
  );
}

function Tess({ r }: { r: Results }) {
  const t = r.tess;
  if (!t) return null;
  const order = ["kepler_as_is", "fine_tuned", "tess_from_scratch", "random_forest_3_features"].filter((k) => t.summary[k]);
  const ft = t.summary["fine_tuned"]?.paired_diff["vs_kepler_as_is"]?.pr_auc;
  const scratch = t.summary["fine_tuned"]?.paired_diff["vs_tess_from_scratch"]?.pr_auc;
  const w = t.kepler_as_is_all_tess;
  return (
    <>
      <Figure
        caption={
          <>
            Da Kepler a TESS: {t.dataset.n_examples} candidati TESS ({t.dataset.n_stars} stelle), {t.n_seeds} divisioni per stella, test di{" "}
            {Math.min(...t.test_sizes)}–{Math.max(...t.test_sizes)} esempi. Media ± deviazione standard.
          </>
        }
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Approccio</th>
                {METRICS.map((m) => (
                  <th key={m.key}>{m.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {order.map((k) => (
                <tr key={k}>
                  <td>{TESS_LABEL[k] ?? k}</td>
                  {METRICS.map((m) => (
                    <td key={m.key}>{pm(t.summary[k].metrics[m.key])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>
      <ul className="notes">
        <li>
          Il modello addestrato solo su Kepler, su tutti i {w.n} candidati TESS, ha ROC-AUC tra {f3(w.roc_auc_range[0])} e{" "}
          {f3(w.roc_auc_range[1])} a seconda del seed: il trasferimento perde rispetto a Kepler ma resta utile.
        </li>
        {ft && scratch && (
          <li>
            Affinarlo su TESS non cambia in modo distinguibile il risultato: PR-AUC {signed(ft.mean)} ± {f3(ft.std)} rispetto al modello così com'è
            ({ft.n_seeds_better} seed su {t.n_seeds}) e {signed(scratch.mean)} ± {f3(scratch.std)} rispetto a una rete addestrata solo su TESS.
          </li>
        )}
      </ul>
    </>
  );
}

function Vetting({ demo }: { demo: Demo }) {
  const v = demo.results.vetting;
  if (!v) return null;
  const models = Object.keys(v.prompt_v2);
  return (
    <>
      <Figure
        caption={`Controlli automatici sui testi di due modelli locali (${v.n_candidates} candidati ciascuno): quota di schede con almeno un difetto noto. Cambia solo il modo in cui i dati vengono presentati al modello.`}
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modello</th>
                <th>Dati come JSON</th>
                <th>Dati come righe etichettate</th>
                <th>Secondi per scheda (mediana)</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m}>
                  <td>{m}</td>
                  <td>{pct(v.prompt_v1[m]?.flagged_rate)}</td>
                  <td>{pct(v.prompt_v2[m]?.flagged_rate)}</td>
                  <td>{Math.round(v.prompt_v2[m]?.seconds_median)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Figure>
      <p className="muted">
        I controlli automatici sono euristiche: i difetti più gravi (numeri veri usati nel ruolo sbagliato, interpretazioni inventate) li ho cercati
        leggendo a mano le schede, e il modello scelto è quello che nella lettura non ne ha mostrati. Le note sono nel repository.
      </p>
    </>
  );
}

export function ResultsSection({ demo }: { demo: Demo }) {
  const r = demo.results;
  return (
    <section id="risultati">
      <h2>Cosa funziona e cosa no</h2>
      <h3>Contro una baseline semplice</h3>
      <ComparisonTable r={r} />
      <h3>Dove il modello sbaglia</h3>
      <Regimes r={r} />
      <h3>Quanto è piccolo il segnale che riconosce</h3>
      <InjectionGrid r={r} />
      <p className="muted">
        L&apos;effemeride è nota, quindi si misura il classificatore, non la ricerca del periodo; i transiti iniettati sono tutti pianeti per
        costruzione, quindi questa è una completezza e non un tasso di falsi positivi.
      </p>
      <h3>Con meno dati o più rumore</h3>
      <Robustness r={r} />
      <h3>Calibrazione e soglia</h3>
      <Calibration r={r} />
      {r.tess && (
        <>
          <h3>Su un altro telescopio: TESS</h3>
          <Tess r={r} />
        </>
      )}
      {r.vetting && (
        <>
          <h3>Le schede di testo</h3>
          <Vetting demo={demo} />
        </>
      )}
    </section>
  );
}
