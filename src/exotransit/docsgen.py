"""Generazione dei documenti (README, model card, post) dai risultati in docs/results/.

Nei modelli di testo (docs/templates/*.tmpl) i numeri sono segnaposto `{{nome}}`: li riempie
`build_values` leggendo gli snapshot dei report, quindi nessun numero e' scritto a mano.
Un segnaposto sconosciuto e' un errore (meglio nessun documento che un numero inventato).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


def render(template: str, values: dict[str, str]) -> str:
    """Sostituisce ogni `{{nome}}`; solleva KeyError con l'elenco dei nomi mancanti."""
    missing = sorted({n for n in _PLACEHOLDER.findall(template) if n not in values})
    if missing:
        raise KeyError(f"Segnaposto senza valore: {missing}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def md_table(header: list[str], rows: list[list[str]], align_right_from: int = 1) -> str:
    """Tabella Markdown; le colonne da `align_right_from` in poi sono allineate a destra."""
    sep = [("---:" if i >= align_right_from else "---") for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def f3(v) -> str:
    return "n/d" if v is None or v != v else f"{v:.3f}"


def f2(v) -> str:
    return "n/d" if v is None or v != v else f"{v:.2f}"


def pm(m: dict) -> str:
    return f"{f3(m['mean'])} ± {f3(m['std'])}"


def signed(v: float, digits: int = 3) -> str:
    return f"{'+' if v >= 0 else '−'}{abs(v):.{digits}f}"


def pct(v: float) -> str:
    return f"{round(v * 100)}%"


def bin_label(bin_: str, index: int, last: int) -> str:
    """ "(10.0, 30.0]" -> "10–30"; primo intervallo "fino a", ultimo "oltre"."""
    m = re.match(r"[\[(]\s*(-?[\d.eE+]+)\s*,\s*(-?[\d.eE+]+)\s*[\])]", bin_)
    if not m:
        return bin_

    def num(s: str) -> str:
        v = float(s)
        return f"{v:.0f}" if abs(v) >= 100 else f"{round(v, 2):g}"

    if index == 0:
        return f"fino a {num(m.group(2))}"
    if index == last:
        return f"oltre {num(m.group(1))}"
    return f"{num(m.group(1))}–{num(m.group(2))}"


def _load(d: Path, name: str):
    return json.loads((d / name).read_text(encoding="utf-8"))


METRICS = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("precision_at_recall90", "Precisione a recall 90%"),
)
VARIANT_LABEL = {
    "no_scalar": "CNN solo curve",
    "full": "CNN completa (curve + feature del catalogo)",
    "local_only": "CNN solo vista locale",
    "global_only": "CNN solo vista globale",
}
TESS_LABEL = {
    "kepler_as_is": "Modello Kepler così com'è",
    "fine_tuned": "Modello Kepler affinato su TESS",
    "tess_from_scratch": "CNN addestrata solo su TESS",
    "random_forest_3_features": "Random forest (3 feature)",
}
ROBUST_ORDER = ["originale", "primi 730 d", "primi 365 d", "rumore x2", "rumore x4"]


def _regime_table(rows: list[dict], title: str) -> str:
    body = [
        [
            bin_label(b["bin"], i, len(rows) - 1),
            b["n"],
            b["n_pos"],
            f2(b["recall"]),
            f2(b["precision"]),
        ]
        for i, b in enumerate(rows)
    ]
    return md_table([title, "Esempi", "Positivi", "Recall", "Precisione"], body)


def build_values(results_dir: str | Path) -> dict[str, str]:
    """Tutti i segnaposto (numeri e tabelle Markdown) calcolati dagli snapshot dei report."""
    d = Path(results_dir)
    ds, cnn = _load(d, "kepler_dataset.json"), _load(d, "kepler_cnn_summary.json")
    base, err = _load(d, "kepler_baselines.json"), _load(d, "kepler_errors.json")
    inj, rob = _load(d, "kepler_injection.json"), _load(d, "kepler_robustness.json")
    shuf = _load(d, "kepler_shuffled_summary.json")
    tess_ds, tess = _load(d, "tess_dataset.json"), _load(d, "tess_transfer_summary.json")
    tess_shuf, vet = _load(d, "tess_shuffled_summary.json"), _load(d, "vetting_summary.json")

    v: dict[str, str] = {}
    # dataset Kepler
    v["kepler_examples"] = str(ds["n_examples"])
    v["kepler_stars"] = str(ds["n_stars"])
    v["kepler_pos"] = str(ds["examples_per_class"]["positivi"])
    v["kepler_neg"] = str(ds["examples_per_class"]["falsi_positivi"])
    v["kepler_multi_stars"] = str(ds["n_stars_with_multiple_kois"])
    v["n_seeds"] = str(len(cnn["seeds"]))
    v["test_size_range"] = f"{min(cnn['test_sizes'])}–{max(cnn['test_sizes'])}"
    v["epochs_max"] = str(cnn["epochs"])
    v["lr"] = f"{cnn['lr']:g}"
    v["batch_size"] = str(cnn["batch_size"])
    demo = _load(d, "demo_meta.json")
    v["demo_n"] = str(demo["n_examples_shown"])
    v["torch_version"] = cnn["versions"]["torch"]
    v["device"] = cnn["device"]
    v["params_no_scalar"] = f"{cnn['n_params']['no_scalar']:,}".replace(",", ".")

    # CNN e baseline (media ± std sui seed)
    s, rf = cnn["summary"], cnn["baseline_random_forest_scalar"]
    for key, label in (
        ("rf", None),
        ("cnn", "no_scalar"),
        ("full", "full"),
        ("local", "local_only"),
        ("global", "global_only"),
    ):
        src = rf if label is None else s[label]["metrics"]
        for m, _ in METRICS:
            v[f"{key}_{m}"] = pm(src[m])
    diff = s["no_scalar"]["paired_diff_vs_rf"]["pr_auc"]
    v["cnn_vs_rf_pr"] = f"{signed(diff['mean'])} ± {f3(diff['std'])}"
    v["cnn_vs_rf_wins"] = f"{diff['n_seeds_better']} su {len(cnn['seeds'])}"
    sn = base["models"]["snr_only"]["test"]
    v["snr_only_roc"] = f3(sn["roc_auc"])
    v["snr_only_roc_ci"] = f"{f3(sn['ci95']['roc_auc'][0])}–{f3(sn['ci95']['roc_auc'][1])}"
    v["shuffle_cnn_roc"] = pm(shuf["summary"]["no_scalar"]["metrics"]["roc_auc"])
    v["shuffle_rf_roc"] = pm(shuf["baseline_random_forest_scalar"]["roc_auc"])

    # errori, calibrazione
    c = err["calibration"]
    v["ece_raw"], v["ece_iso"] = pm(c["ece_raw"]), pm(c["ece_isotonic"])
    v["brier_raw"], v["brier_iso"] = pm(c["brier_raw"]), pm(c["brier_isotonic"])
    for t in ("0.9", "0.95"):
        key = t.replace(".", "")
        v[f"target{key}_value"] = f"{float(t):.2f}"
        v[f"target{key}_recall"] = pm(c[f"target{t}_test_recall"])
        v[f"target{key}_precision"] = pm(c[f"target{t}_test_precision"])
    v["regimes_rows"], v["regimes_kois"] = str(err["n_rows"]), str(err["n_unique_kois"])
    per, dep, snr = (
        err["tables"]["periodo_giorni"],
        err["tables"]["profondita_ppm"],
        err["tables"]["snr_catalogo"],
    )
    v["period_short_recall"] = f2(per[0]["recall"])
    v["period_short_pos"] = str(per[0]["n_pos"])
    v["period_short_label"] = bin_label(per[0]["bin"], 0, len(per) - 1)
    v["depth_deep_recall"] = f2(dep[-1]["recall"])
    v["depth_deep_pos"] = str(dep[-1]["n_pos"])
    v["depth_deep_n"] = str(dep[-1]["n"])
    v["depth_deep_label"] = bin_label(dep[-1]["bin"], len(dep) - 1, len(dep) - 1)
    mid = [b["recall"] for b in per[2:5]]
    lo = re.match(r"[\[(]\s*(-?[\d.]+)", per[2]["bin"]).group(1)
    hi = re.search(r",\s*(-?[\d.]+)\s*[\])]", per[4]["bin"]).group(1)
    v["period_mid_range"] = f"{float(lo):g} e {float(hi):g}"
    v["period_mid_recall"] = f"{f2(min(mid))}–{f2(max(mid))}"
    v["snr_low_recall"] = f2(snr[1]["recall"])
    v["snr_low_label"] = bin_label(snr[1]["bin"], 1, len(snr) - 1)

    # injection
    cells = [x for x in inj["cells"] if x["mean"] is not None]
    worst = min(cells, key=lambda x: x["mean"])
    v["inj_n"], v["inj_valid"] = str(inj["n_injections"]), str(inj["n_valid"])
    v["inj_worst"] = f2(worst["mean"])
    v["inj_worst_cell"] = f"{worst['depth_ppm']} ppm e {worst['period']} giorni"
    v["inj_median"] = f2(sorted(x["mean"] for x in cells)[len(cells) // 2])

    # robustezza
    o = rob["originale"]
    v["rob_orig_pr"], v["rob_orig_recall"] = pm(o["pr_auc"]), pm(o["recall"])
    for key, name in (
        ("365", "primi 365 d"),
        ("730", "primi 730 d"),
        ("noise2", "rumore x2"),
        ("noise4", "rumore x4"),
    ):
        v[f"rob_{key}_recall"] = pm(rob[name]["recall"])
        v[f"rob_{key}_pr"] = pm(rob[name]["pr_auc"])
        v[f"rob_{key}_kept"] = (
            f"{round(rob[name]['n_kept_mean'])}/{round(rob[name]['n_test_mean'])}"
        )

    # TESS
    ts = tess["summary"]
    v["tess_examples"], v["tess_stars"] = str(tess_ds["n_examples"]), str(tess_ds["n_stars"])
    v["tess_pos"] = str(tess_ds["examples_per_class"]["positivi"])
    v["tess_neg"] = str(tess_ds["examples_per_class"]["falsi_positivi"])
    v["tess_test_range"] = f"{min(tess['test_sizes'])}–{max(tess['test_sizes'])}"
    for key, name in (
        ("asis", "kepler_as_is"),
        ("ft", "fine_tuned"),
        ("scratch", "tess_from_scratch"),
        ("rf", "random_forest_3_features"),
    ):
        for m, _ in METRICS:
            v[f"tess_{key}_{m}"] = pm(ts[name]["metrics"][m])
    ft = ts["fine_tuned"]["paired_diff"]
    a, b = ft["vs_kepler_as_is"]["pr_auc"], ft["vs_tess_from_scratch"]["pr_auc"]
    v["tess_ft_vs_asis"] = f"{signed(a['mean'])} ± {f3(a['std'])}"
    v["tess_ft_vs_asis_wins"] = f"{a['n_seeds_better']} su {tess['n_seeds']}"
    v["tess_ft_vs_scratch"] = f"{signed(b['mean'])} ± {f3(b['std'])}"
    w = tess["kepler_as_is_all_tess"]
    v["tess_whole_n"] = str(w["n"])
    v["tess_whole_roc_range"] = f"{f3(w['roc_auc_range'][0])}–{f3(w['roc_auc_range'][1])}"
    v["tess_shuffle_roc"] = pm(tess_shuf["summary"]["kepler_as_is"]["roc_auc"])

    # schede di testo
    v["vet_n"] = str(vet["n_candidates"])
    for key, model in (("gemma", "gemma3:12b"), ("qwen", "qwen2.5-coder:7b")):
        v[f"vet_{key}_v1"] = pct(vet["prompt_v1"][model]["flagged_rate"])
        v[f"vet_{key}_v2"] = pct(vet["prompt_v2"][model]["flagged_rate"])
        v[f"vet_{key}_seconds"] = str(round(vet["prompt_v2"][model]["seconds_median"]))

    # tabelle Markdown
    v["table_comparison"] = md_table(
        ["Modello", *[lab for _, lab in METRICS]],
        [["Random forest (4 feature del catalogo)", *[pm(rf[m]) for m, _ in METRICS]]]
        + [
            [VARIANT_LABEL[k], *[pm(s[k]["metrics"][m]) for m, _ in METRICS]]
            for k in VARIANT_LABEL
            if k in s
        ],
    )
    v["table_regimes_period"] = _regime_table(per, "Periodo (giorni)")
    v["table_regimes_depth"] = _regime_table(dep, "Profondità (ppm)")
    v["table_regimes_snr"] = _regime_table(snr, "SNR del catalogo")
    periods, depths = inj["periods"], inj["depths_ppm"]
    at = {(x["depth_ppm"], x["period"]): x["mean"] for x in inj["cells"]}
    v["table_injection"] = md_table(
        ["Profondità \\ periodo (giorni)", *[str(p) for p in periods]],
        [[f"{dp} ppm", *[f2(at.get((dp, p))) for p in periods]] for dp in depths],
    )
    v["table_robustness"] = md_table(
        ["Perturbazione", "KOI tenuti", "PR-AUC", "Recall", "Precisione"],
        [
            [
                k.capitalize(),
                v_kept,
                pm(rob[k]["pr_auc"]),
                pm(rob[k]["recall"]),
                f3(rob[k]["precision"]["mean"]),
            ]
            for k in ROBUST_ORDER
            if k in rob
            for v_kept in [f"{round(rob[k]['n_kept_mean'])}/{round(rob[k]['n_test_mean'])}"]
        ],
    )
    v["table_tess"] = md_table(
        ["Approccio", *[lab for _, lab in METRICS]],
        [
            [TESS_LABEL[k], *[pm(ts[k]["metrics"][m]) for m, _ in METRICS]]
            for k in TESS_LABEL
            if k in ts
        ],
    )
    v["table_vetting"] = md_table(
        [
            "Modello",
            "Dati come JSON",
            "Dati come righe etichettate",
            "Secondi per scheda (mediana)",
        ],
        [
            [
                m,
                pct(vet["prompt_v1"][m]["flagged_rate"]),
                pct(vet["prompt_v2"][m]["flagged_rate"]),
                str(round(vet["prompt_v2"][m]["seconds_median"])),
            ]
            for m in vet["prompt_v2"]
        ],
    )
    return v
