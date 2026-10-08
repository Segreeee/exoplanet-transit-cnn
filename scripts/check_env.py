"""Controlla l'ambiente: librerie, GPU, Ollama. Esegui: python scripts/check_env.py

Codice di uscita 1 solo se manca qualcosa di essenziale (librerie o pacchetto exotransit).
GPU e Ollama sono facoltativi: senza GPU si addestra su CPU (lento), senza Ollama si usa il
testo di riserva per le schede di vetting.
"""

from __future__ import annotations

import importlib
import platform
import sys

ESSENTIAL = ["numpy", "scipy", "pandas", "astropy", "sklearn", "requests", "torch", "exotransit"]
OPTIONAL = {"lightkurve": "serve per scaricare Kepler/TESS (pip install -e '.[kepler]')"}


def check_libs() -> bool:
    ok = True
    for name in ESSENTIAL:
        try:
            mod = importlib.import_module(name)
            print(f"  OK   {name} {getattr(mod, '__version__', '')}")
        except Exception as exc:
            ok = False
            print(f"  MANCA {name}: {type(exc).__name__}: {exc}")
    for name, why in OPTIONAL.items():
        try:
            importlib.import_module(name)
            print(f"  OK   {name}")
        except Exception:
            print(f"  --   {name} non installato ({why})")
    return ok


def check_gpu() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        print("  ATTENZIONE: CUDA non disponibile, l'addestramento girera' su CPU.")
        print("  Controlla `nvidia-smi` e di aver installato la build di PyTorch con CUDA")
        print("  (selettore su https://pytorch.org/get-started/locally/).")
        return
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1024**3
    print(f"  OK   {p.name}, {vram:.1f} GB di VRAM, torch CUDA {torch.version.cuda}")


def check_ollama() -> None:
    from exotransit.llm import OllamaClient

    client = OllamaClient(timeout=5.0)
    print(f"  Host: {client.host} | OLLAMA_MODEL: {client.model or '(non impostato)'}")
    try:
        models = client.list_models()
    except Exception as exc:
        print(f"  Ollama non raggiungibile ({type(exc).__name__}). Avvialo con `ollama serve`.")
        return
    print(f"  OK   modelli installati: {', '.join(models) if models else '(nessuno)'}")
    if client.model and client.model not in models:
        print(f"  ATTENZIONE: {client.model!r} non e' tra i modelli installati.")


def main() -> int:
    print(f"Python {sys.version.split()[0]} su {platform.platform()}")
    print("Librerie:")
    ok = check_libs()
    print("GPU:")
    check_gpu()
    print("Ollama:")
    try:
        check_ollama()
    except Exception as exc:
        print(f"  Controllo saltato: {type(exc).__name__}: {exc}")
    print("\nAmbiente OK." if ok else "\nCi sono librerie mancanti: vedi README, sezione Setup.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
