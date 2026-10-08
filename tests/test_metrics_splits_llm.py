import numpy as np
import pytest
import requests

from exotransit.llm import (
    OllamaClient,
    candidate_facts,
    numbers_not_in_source,
    template_report,
    vetting_report,
)
from exotransit.metrics import best_f1_threshold, evaluate_scores, precision_at_recall
from exotransit.preprocess import Ephemeris
from exotransit.splits import assert_no_leakage, split_by_group

# ---------- metriche ----------


def test_metrics_perfect_and_inverted():
    y = np.array([0, 0, 1, 1, 0, 1])
    perfect = evaluate_scores(y, y.astype(float))
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert perfect["precision"] == 1.0 and perfect["recall"] == 1.0
    inverted = evaluate_scores(y, 1.0 - y)
    assert inverted["roc_auc"] == 0.0 and inverted["recall"] == 0.0


def test_metrics_single_class_is_nan_not_crash():
    out = evaluate_scores(np.zeros(5), np.linspace(0, 1, 5))
    assert np.isnan(out["roc_auc"]) and np.isnan(out["pr_auc"])


def test_precision_at_recall():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert precision_at_recall(y, s, 0.9) == 1.0
    assert precision_at_recall(y, np.array([0.1, 0.2, 0.9, 0.8]), 0.99) == pytest.approx(0.5)


# ---------- split per gruppo ----------


def _toy(n_groups=120, per_group=2, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    y = rng.random(groups.size) < 0.3
    return y.astype(int), groups


def test_split_by_group_no_overlap_and_covers_everything():
    y, groups = _toy()
    s = split_by_group(y, groups, seed=1)
    allidx = np.concatenate([s["train"], s["val"], s["test"]])
    assert len(allidx) == len(np.unique(allidx)) == len(y)
    assert_no_leakage(s, groups)
    assert 0.05 < len(s["test"]) / len(y) < 0.2
    assert abs(y[s["test"]].mean() - y.mean()) < 0.15  # classi circa stratificate


def test_split_is_reproducible():
    y, groups = _toy()
    a, b = split_by_group(y, groups, seed=5), split_by_group(y, groups, seed=5)
    np.testing.assert_array_equal(a["test"], b["test"])


def test_leakage_guard_catches_shared_group():
    groups = np.array([0, 0, 1, 1])
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_leakage({"train": np.array([0, 2]), "test": np.array([1, 3])}, groups)


def test_split_requires_enough_groups():
    with pytest.raises(ValueError):
        split_by_group(np.array([0, 1] * 3), np.arange(6), n_folds=10)


# ---------- Ollama (con finto server) ----------


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, reply="ok", status=200):
        self.reply, self.status, self.calls = reply, status, []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        return FakeResponse({"models": [{"name": "m1:7b"}, {"name": "m2:3b"}]}, self.status)

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return FakeResponse({"message": {"content": self.reply}}, self.status)


EPH = Ephemeris(period=5.1234, t0=1.0, duration=0.1, depth=0.0021, snr=14.37)


def test_ollama_client_endpoints_and_host_normalisation():
    s = FakeSession("ciao")
    c = OllamaClient(host="127.0.0.1:11434", model="m1:7b", session=s)
    assert c.host == "http://127.0.0.1:11434"
    assert c.list_models() == ["m1:7b", "m2:3b"]
    assert c.chat("domanda", system="sys") == "ciao"
    _, url, payload = s.calls[-1]
    assert url.endswith("/api/chat") and payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"


def test_ollama_requires_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        OllamaClient(session=FakeSession()).chat("x")


def test_candidate_facts_units():
    f = candidate_facts("KOI-X", EPH, prob_planet=0.9123)
    assert f["durata_ore"] == 2.4 and f["profondita_ppm"] == 2100
    assert f["esito_classificatore"] == "candidato"
    assert candidate_facts("KOI-X", EPH, 0.2)["esito_classificatore"] == "scartato"


def test_number_check_flags_invented_values_only():
    facts = candidate_facts("KOI-X", EPH, 0.91)
    honest = "Periodo 5.1234 giorni, durata 2.4 ore, 2100 ppm, SNR 14.4, probabilita' 0.91."
    assert numbers_not_in_source(honest, facts) == []
    invented = honest + " Raggio stimato 1.8 raggi terrestri e massa 7.7."
    assert set(numbers_not_in_source(invented, facts)) == {"1.8", "7.7"}


def test_vetting_report_uses_llm_and_flags_numbers():
    facts = candidate_facts("KOI-X", EPH, 0.91)
    s = FakeSession("Periodo 5.1234 giorni; pianeta di 99.5 masse terrestri.")
    out = vetting_report(OllamaClient(model="m", session=s), facts)
    assert out["used_llm"] and out["suspicious_numbers"] == ["99.5"]


def test_vetting_report_falls_back_when_ollama_down():
    facts = candidate_facts("KOI-X", EPH, 0.91)
    out = vetting_report(OllamaClient(model="m", session=FakeSession(status=500)), facts)
    assert not out["used_llm"] and out["text"] == template_report(facts)
    assert "HTTPError" in out["error"]
    # nessun modello configurato: stesso comportamento, nessun crash
    out2 = vetting_report(OllamaClient(model=None, session=FakeSession()), facts)
    assert not out2["used_llm"]


def test_best_f1_threshold_handles_uncalibrated_scores():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.01, 0.02, 0.03, 0.05, 0.10, 0.12, 0.20, 0.30])  # ranking perfetto
    assert evaluate_scores(y, score, threshold=0.5)["recall"] == 0.0
    t = best_f1_threshold(y, score)
    assert 0.05 < t <= 0.10
    assert evaluate_scores(y, score, threshold=t)["recall"] == 1.0


def test_best_f1_threshold_single_class_returns_default():
    assert best_f1_threshold(np.zeros(5), np.linspace(0, 1, 5), default=0.42) == 0.42
