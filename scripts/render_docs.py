"""Genera README.md, docs/MODEL_CARD.md e docs/POST.md dai modelli in docs/templates/.

    python scripts/render_docs.py            # riscrive i file
    python scripts/render_docs.py --check    # esce con 1 se i file non sono aggiornati (per la CI)

I numeri vengono da docs/results/*.json (vedi snapshot_results.py): nessuno e' scritto a mano.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exotransit.docsgen import build_values, render

TARGETS = {
    "docs/templates/README.md.tmpl": "README.md",
    "docs/templates/MODEL_CARD.md.tmpl": "docs/MODEL_CARD.md",
    "docs/templates/POST.md.tmpl": "docs/POST.md",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", default="docs/results")
    ap.add_argument(
        "--check", action="store_true", help="non scrive: verifica che i file siano aggiornati"
    )
    args = ap.parse_args()

    values = build_values(args.results)
    stale = []
    for tmpl, out in TARGETS.items():
        text = render(Path(tmpl).read_text(encoding="utf-8"), values)
        path = Path(out)
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else None
            if current != text:
                stale.append(out)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"scritto {out} ({len(text.splitlines())} righe)")
    if stale:
        print(f"File non aggiornati: {stale}. Esegui: python scripts/render_docs.py")
        return 1
    if args.check:
        print("Documenti aggiornati.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
