"""Schede di vetting in linguaggio naturale con un LLM locale (Ollama).

Ruolo dell'LLM: SOLO riformulare in italiano numeri che la pipeline ha gia' calcolato.
Non classifica, non calcola, non "interpreta" la curva di luce. Un controllo euristico
(`numbers_not_in_source`) segnala i numeri del testo che non compaiono nei dati forniti,
e se Ollama non risponde si ricade su un testo deterministico (`template_report`).
"""

from __future__ import annotations

import os
import re
import statistics
from typing import Any

import requests

from exotransit.preprocess import Ephemeris

SYSTEM_PROMPT = (
    "Sei un assistente che redige brevi schede di vetting per candidati esopianeta. "
    "Usa ESCLUSIVAMENTE i dati forniti, ognuno con la sua etichetta: non inventare numeri, nomi "
    "di stelle o di pianeti, ne' conclusioni astrofisiche non supportate dai dati. Non "
    "menzionare dati che non compaiono (massa, raggio, atmosfera e simili). Non confondere "
    "durata e periodo; non confrontare mai l'SNR con la soglia di decisione, che si applica "
    "solo alla probabilita'. Non dare consigli o raccomandazioni su ulteriori indagini. Usa solo "
    "la parola 'candidato', mai 'pianeta', 'scoperto' o 'confermato'. La probabilita' e' un "
    "punteggio di un classificatore, non una conferma. Riporta l'esito del classificatore. "
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
        "snr": round(eph.snr, 1),
        "probabilita_pianeta": round(prob_planet, 3),
        "soglia_decisione": threshold,
        "esito_classificatore": "candidato" if prob_planet >= threshold else "scartato",
    }


def template_report(facts: dict) -> str:
    """Testo deterministico di riserva (nessun LLM)."""
    return (
        f"Candidato {facts['candidato']}: periodo {facts['periodo_giorni']} giorni, durata "
        f"{facts['durata_ore']} ore, profondita' {facts['profondita_ppm']} ppm, SNR "
        f"{facts['snr']}. Il classificatore assegna probabilita' "
        f"{facts['probabilita_pianeta']} (soglia {facts['soglia_decisione']}): "
        f"esito '{facts['esito_classificatore']}'. "
        "Il punteggio non e' una conferma: serve verifica indipendente."
    )


def format_facts(facts: dict) -> str:
    """Dati del candidato come righe etichettate (piu' difficili da scambiare di un JSON con chiavi
    abbreviate: con il JSON gli LLM locali confondevano durata e periodo, SNR e soglia)."""
    return "\n".join(
        [
            f"Candidato: {facts['candidato']}",
            f"Periodo orbitale: {facts['periodo_giorni']} giorni",
            f"Durata del transito: {facts['durata_ore']} ore (non e' il periodo)",
            f"Profondita' del transito: {facts['profondita_ppm']} ppm",
            f"SNR (rapporto segnale/rumore): {facts['snr']} (NON si confronta con la soglia)",
            f"Probabilita' del classificatore: {facts['probabilita_pianeta']}",
            f"Soglia di decisione del classificatore: {facts['soglia_decisione']} "
            "(si applica solo alla probabilita')",
            f"Esito del classificatore: {facts['esito_classificatore']}",
        ]
    )


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _source_numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, str):  # es. il nome "K00301.01": le sue cifre sono nei dati
        return [float(t.replace(",", ".")) for t in _NUMBER.findall(obj)]
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
    allowed += [a * 100 for a in allowed]  # una probabilita' 0.93 puo' diventare "93%"
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
    user = "Dati del candidato:\n" + format_facts(facts)
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


_FORBIDDEN = re.compile(
    r"\bscopert[oa]\b|\bscoperta\b|\b(?:esopianeta|pianeta)\s+confermat[oa]\b"
    r"|\b(?:e'|\u00e8|e)\s+un\s+(?:esopianeta|esoplaneta|pianeta)\b",
    re.IGNORECASE,
)
# difetti semantici visti con qwen2.5-coder:7b e gemma3:12b: numeri VALIDI nel ruolo sbagliato.
# Euristiche tarate sui testi osservati: non e' detto che generalizzino. ". " chiude la frase.
_SNR_SOGLIA = re.compile(r"\bsnr\b(?:(?!\.\s)[^\n]){0,60}?\bsoglia", re.IGNORECASE)
_PERIODO_ORE = re.compile(
    r"\bgiorni(?:(?!\.\s)[^\n]){0,30}?corrispond\w+\s+a\s+(?:una\s+durata\s+di\s+)?\d+(?:[.,]\d+)?\s*ore",
    re.IGNORECASE,
)
_RACCOMANDAZIONI = re.compile(
    r"\bulterior[ie]\s+(?:indagin\w+|analis\w+|osservazion\w+|verific\w+|approfondiment\w+)",
    re.IGNORECASE,
)
MAX_WORDS = 120


def check_report(text: str, facts: dict) -> list[str]:
    """Controlli automatici su una scheda (vuota = nessun problema rilevato):
    numeri non presenti nei dati, oltre 120 parole, linguaggio da scoperta/conferma (regola 6:
    'candidato', mai 'pianeta scoperto'), esito non riportato e tre difetti semantici osservati
    (SNR confrontato con la soglia, durata scambiata col periodo, raccomandazioni non supportate).
    Euristiche: non provano che il testo sia corretto, solo che non ha difetti noti."""
    flags = []
    if numbers_not_in_source(text, facts):
        flags.append("numeri_sospetti")
    if len(text.split()) > MAX_WORDS:
        flags.append("troppo_lungo")
    if _FORBIDDEN.search(text):
        flags.append("linguaggio_non_ammesso")
    if _SNR_SOGLIA.search(text):
        flags.append("snr_confuso_con_soglia")
    if _PERIODO_ORE.search(text):
        flags.append("periodo_durata_confusi")
    if _RACCOMANDAZIONI.search(text):
        flags.append("raccomandazioni_non_supportate")
    stem = "scartat" if facts["esito_classificatore"] == "scartato" else "candidat"
    if stem not in text.lower():
        flags.append("esito_non_riportato")
    return flags


def summarize_reports(records: list[dict]) -> dict:
    """Riepilogo di un modello su N schede. Ogni record: used_llm, flags, seconds, words.

    Il tasso di fallback e' su tutte le schede; flag, tempi e lunghezza sulle sole schede
    prodotte dall'LLM (il template deterministico non e' un risultato del modello).
    """
    llm = [r for r in records if r["used_llm"]]
    flag_names = sorted({f for r in llm for f in r["flags"]})
    sec = sorted(r["seconds"] for r in llm)
    n = max(len(llm), 1)
    return {
        "n": len(records),
        "n_llm": len(llm),
        "fallback_rate": 1 - len(llm) / max(len(records), 1),
        "flagged_rate": sum(bool(r["flags"]) for r in llm) / n,
        "flag_rates": {f: sum(f in r["flags"] for r in llm) / n for f in flag_names},
        "seconds_median": float(statistics.median(sec)) if sec else float("nan"),
        "seconds_p95": float(sec[min(int(0.95 * len(sec)), len(sec) - 1)]) if sec else float("nan"),
        "words_mean": sum(r["words"] for r in llm) / n if llm else float("nan"),
    }
