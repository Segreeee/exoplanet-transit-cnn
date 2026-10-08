"""Schede di vetting in linguaggio naturale con un LLM locale (Ollama).

Ruolo dell'LLM: SOLO riformulare in italiano numeri che la pipeline ha gia' calcolato.
Non classifica, non calcola, non "interpreta" la curva di luce. Un controllo euristico
(`numbers_not_in_source`) segnala i numeri del testo che non compaiono nei dati forniti,
e se Ollama non risponde si ricade su un testo deterministico (`template_report`).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from exotransit.preprocess import Ephemeris

SYSTEM_PROMPT = (
    "Sei un assistente che redige brevi schede di vetting per candidati esopianeta. "
    "Usa ESCLUSIVAMENTE i dati JSON forniti. Non inventare numeri, nomi di stelle o di pianeti, "
    "ne' conclusioni astrofisiche non supportate dai dati. Se un dato manca scrivi 'non "
    "disponibile'. La probabilita' e' un punteggio di un classificatore, non una conferma. "
    "Rispondi in italiano, in al massimo 120 parole, senza elenchi puntati."
)


class OllamaClient:
    """Client minimale per l'API HTTP di Ollama (/api/tags e /api/chat)."""

    def __init__(self, host=None, model=None, timeout: float = 120.0, session=None):
        host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        self.host = host.rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL")
        self.timeout = timeout
        self.session = session or requests.Session()

    def list_models(self) -> list[str]:
        r = self.session.get(f"{self.host}/api/tags", timeout=self.timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def chat(self, user: str, system: str | None = None, temperature: float = 0.1) -> str:
        if not self.model:
            raise RuntimeError("Nessun modello: imposta OLLAMA_MODEL o passa model=...")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


def candidate_facts(name: str, eph: Ephemeris, prob_planet: float, threshold: float = 0.5) -> dict:
    """Dati del candidato gia' convertiti nelle unita' che finiranno nel testo, cosi' l'LLM
    non deve fare calcoli."""
    return {
        "candidato": name,
        "periodo_giorni": round(eph.period, 4),
        "durata_ore": round(eph.duration * 24, 2),
        "profondita_ppm": round(eph.depth * 1e6),
        "snr_bls": round(eph.snr, 1),
        "probabilita_pianeta": round(prob_planet, 3),
        "soglia_decisione": threshold,
        "esito_classificatore": "candidato" if prob_planet >= threshold else "scartato",
    }


def template_report(facts: dict) -> str:
    """Testo deterministico di riserva (nessun LLM)."""
    return (
        f"Candidato {facts['candidato']}: periodo {facts['periodo_giorni']} giorni, durata "
        f"{facts['durata_ore']} ore, profondita' {facts['profondita_ppm']} ppm, SNR BLS "
        f"{facts['snr_bls']}. Il classificatore assegna probabilita' "
        f"{facts['probabilita_pianeta']} (soglia {facts['soglia_decisione']}): "
        f"esito '{facts['esito_classificatore']}'. "
        "Il punteggio non e' una conferma: serve verifica indipendente."
    )


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _source_numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _source_numbers(v)]
    if isinstance(obj, list | tuple):
        return [x for v in obj for x in _source_numbers(v)]
    return []


def numbers_not_in_source(report: str, facts: dict, rel_tol: float = 0.02) -> list[str]:
    """Numeri del testo che non corrispondono a nessun valore dei dati (tolleranza relativa
    2% + arrotondamento). Euristica: ignora gli interi di una sola cifra (elenchi, "3 ore")
    e non intercetta errori che riusano numeri validi nel posto sbagliato."""
    allowed = _source_numbers(facts)
    allowed += [a * 24 for a in allowed] + [a / 24 for a in allowed]  # tolleranza conversioni
    suspicious = []
    for token in _NUMBER.findall(report):
        value = float(token.replace(",", "."))
        if token.isdigit() and len(token) == 1:
            continue
        if not any(abs(value - a) <= max(rel_tol * abs(a), 0.06) for a in allowed):
            suspicious.append(token)
    return suspicious


def vetting_report(client: OllamaClient, facts: dict) -> dict:
    """Scheda di vetting: prova l'LLM, controlla i numeri, altrimenti usa il template."""
    user = "Dati del candidato (JSON):\n" + json.dumps(facts, ensure_ascii=False, indent=2)
    try:
        text = client.chat(user, system=SYSTEM_PROMPT).strip()
    except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
        return {
            "text": template_report(facts),
            "used_llm": False,
            "error": f"{type(exc).__name__}: {exc}",
            "suspicious_numbers": [],
        }
    return {
        "text": text,
        "used_llm": True,
        "error": None,
        "suspicious_numbers": numbers_not_in_source(text, facts),
    }
