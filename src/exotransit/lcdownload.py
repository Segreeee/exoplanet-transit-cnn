"""Helper di download delle curve di luce (lightkurve), condivisi tra Kepler e TESS.

Nessuna rete qui: solo gestione di cache, file parziali e spazio su disco, cosi' e' testabile.
Problemi reali incontrati con MAST/lightkurve (vedi CLAUDE.md, regola 9):
- un download interrotto lascia FITS da 0 byte in ~/.lightkurve e la stella fallisce per sempre;
- la cache FITS pesa ~14x il .npz salvato: va cancellata stella per stella.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MIN_FREE_BYTES = 3_000_000_000  # i download si fermano sotto 3 GB liberi


def star_cache_dirs(cache_root: Path, star_id: int, mission: str = "Kepler") -> list[Path]:
    """Cartelle della cache lightkurve di una stella (KIC per Kepler, TIC per TESS)."""
    base = cache_root / "mastDownload" / mission
    pattern = {"Kepler": f"kplr{star_id:09d}*", "TESS": f"tess*-{star_id:016d}-*"}[mission]
    return list(base.glob(pattern))


def purge_empty_cache_files(cache_root: Path, star_id: int, mission: str = "Kepler") -> int:
    """Cancella i file da 0 byte della stella nella cache di lightkurve; restituisce quanti."""
    n = 0
    for f in (f for d in star_cache_dirs(cache_root, star_id, mission) for f in d.glob("*")):
        if f.is_file() and f.stat().st_size == 0:
            f.unlink()
            n += 1
    return n


def clear_star_cache(cache_root: Path, star_id: int, mission: str = "Kepler") -> int:
    """Cancella i FITS grezzi della stella; restituisce il numero di cartelle."""
    dirs = star_cache_dirs(cache_root, star_id, mission)
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    return len(dirs)


def has_free_space(path: Path, min_bytes: float, disk_usage=shutil.disk_usage) -> bool:
    return disk_usage(path).free >= min_bytes


def download_with_cache_recovery(found, purge):
    """`download_all`; se fallisce chiama `purge()` (rimuove i FITS troncati) e riprova una volta.

    Un'esecuzione interrotta lascia file da 0 byte in ~/.lightkurve: lightkurve li considera
    scaricati e fallisce per sempre con "Not recognized as a supported data product"
    (anche con `cache=False`, verificato).
    """
    try:
        return found.download_all()
    except Exception:
        purge()
        return found.download_all()


def save_npz_atomic(path: Path, **arrays) -> None:
    """Scrive su file temporaneo e rinomina: un'interruzione non lascia un .npz parziale."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run_downloads(fetch, jobs: list, workers: int, failed_path: str) -> tuple[int, int]:
    """Esegue `fetch(job) -> (stella, errore o None)` in PROCESSI paralleli, stampa l'avanzamento e
    scrive i fallimenti in `failed_path` (stella<TAB>errore). Restituisce (riuscite, fallite).

    Processi e non thread: `lightkurve.utils.suppress_stdout` sostituisce `sys.stdout` globale e con
    piu' thread lo lascia chiuso (verificato: crash). `fetch` deve essere una funzione di modulo.
    """
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (star, err) in enumerate(pool.map(fetch, jobs), 1):
            msg = "ok"
            if err:
                msg = "FALLITA: " + err.splitlines()[0]
                failed.append((star, err.splitlines()[0]))
            print(f"[{i}/{len(jobs)}] {star} {msg}", flush=True)
    Path(failed_path).write_text("".join(f"{k}\t{e}\n" for k, e in failed), encoding="utf-8")
    return len(jobs) - len(failed), len(failed)
