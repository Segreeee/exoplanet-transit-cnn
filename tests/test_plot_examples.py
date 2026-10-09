"""Selezione degli esempi per l'ispezione visiva (scripts/plot_kepler_examples.py)."""

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "plot_kepler_examples", Path(__file__).parents[1] / "scripts" / "plot_kepler_examples.py"
)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)


def _sub():
    return pd.DataFrame(
        {
            "kepid": [1, 1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e", "f"],
            "label": [1, 1, 1, 0, 0, 0],
            "snr": [10.0, 30.0, 20.0, 5.0, 6.0, 7.0],
        }
    )


def test_pick_examples_is_stratified_one_koi_per_star_with_highest_snr():
    picked = pe.pick_examples(_sub(), n=4, seed=0)
    assert len(picked) == 4 and picked["kepid"].is_unique
    assert picked["label"].value_counts().to_dict() == {1: 2, 0: 2}
    assert picked.loc[picked["kepid"] == 1, "name"].item() == "b"  # SNR 30 > 10


def test_pick_examples_is_deterministic_and_handles_small_classes():
    a = pe.pick_examples(_sub(), n=10, seed=3)
    b = pe.pick_examples(_sub(), n=10, seed=3)
    assert list(a["name"]) == list(b["name"])
    assert len(a) == 5  # solo 5 stelle disponibili
