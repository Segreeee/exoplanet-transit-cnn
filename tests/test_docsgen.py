"""Generazione dei documenti dai risultati: segnaposto, tabelle e coerenza con gli snapshot."""

import json
from pathlib import Path

import pytest

from exotransit.docsgen import bin_label, build_values, md_table, render

RESULTS = Path(__file__).parents[1] / "docs" / "results"


def test_render_fills_placeholders_and_rejects_unknown_ones():
    assert render("a {{x}} b {{ y }}", {"x": "1", "y": "2"}) == "a 1 b 2"
    with pytest.raises(KeyError, match="zzz"):
        render("a {{zzz}}", {"x": "1"})


def test_md_table_alignment_and_rows():
    t = md_table(["A", "B"], [["x", 1], ["y", 2]])
    assert t.splitlines()[1] == "| --- | ---: |"
    assert t.splitlines()[2] == "| x | 1 |"


def test_bin_label_formats_intervals():
    assert bin_label("(-0.001, 1.5]", 0, 5) == "fino a 1.5"
    assert bin_label("(10.0, 30.0]", 3, 5) == "10–30"
    assert bin_label("(100.0, 5000.0]", 5, 5) == "oltre 100"


@pytest.mark.skipif(not RESULTS.exists(), reason="snapshot dei risultati assente")
def test_build_values_matches_the_snapshots_and_has_no_empty_value():
    v = build_values(RESULTS)
    cnn = json.loads((RESULTS / "kepler_cnn_summary.json").read_text(encoding="utf-8"))
    mean = cnn["summary"]["no_scalar"]["metrics"]["pr_auc"]["mean"]
    assert v["cnn_pr_auc"].startswith(f"{mean:.3f} ±")
    assert all(str(x).strip() for x in v.values())
    assert v["table_comparison"].count("\n") >= 5  # intestazione, separatore, almeno 4 modelli
    assert "n/d" not in v["table_injection"]
