#!/usr/bin/env python3
"""Export Markdown/LaTeX ablation tables from results/*/metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]

# Ablation ID → results folder (committed metrics preferred)
ABLATIONS = [
    ("A0", "tiny_baseline", "w/o token InfoNCE (easy)"),
    ("A1", "tiny_align_v2", "+ token InfoNCE (easy)"),
    ("A2", "tiny_hard_v1", "hard, no curriculum"),
    ("A3", "tiny_hard_v2", "hard + curriculum"),
    ("A4", "tiny_hard_v3", "hard stronger push"),
    ("A5", "tiny_real_v4", "Flickr+CLIP tiny"),
    ("A6a", "tiny_real_v5", "Flickr+CLIP small"),
    ("A6b", "tiny_real_v6", "Flickr scaled tiny"),
    ("A6c", "tiny_real_coco_v2", "COCO scaled tiny"),
]

# Fallback when metrics.json missing or incomplete (from EXPERIMENTS.md)
FALLBACK = {
    "tiny_baseline": {"alignment_acc": 0.288, "cmce_score": 0.573, "chunk_f1": 1.0},
    "tiny_align_v2": {"alignment_acc": 0.817, "cmce_score": 0.890, "chunk_f1": 1.0},
    "tiny_hard_v1": {"alignment_acc": 0.142, "cmce_score": 0.483, "chunk_f1": 0.99},
    "tiny_hard_v2": {"alignment_acc": 0.356, "cmce_score": 0.607},
    "tiny_hard_v3": {"alignment_acc": 0.380, "cmce_score": 0.612},
    "tiny_real_v4": {"alignment_acc": 0.320, "cmce_score": 0.557, "chunk_f1": 0.912},
    "tiny_real_v5": {"alignment_acc": 0.156, "cmce_score": 0.471, "chunk_f1": 0.945},
    "tiny_real_v6": {"alignment_acc": 0.273, "cmce_score": 0.526, "chunk_f1": 0.906},
    "tiny_real_coco_v2": {"alignment_acc": 0.291, "cmce_score": 0.535, "chunk_f1": 0.900},
}


def load_metrics(folder: str) -> Dict[str, Any]:
    path = ROOT / "results" / folder / "metrics.json"
    data: Dict[str, Any] = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # normalize nested "metrics" dict
        nested = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        data = {**FALLBACK.get(folder, {}), **nested, **raw}
    else:
        data = dict(FALLBACK.get(folder, {}))
    return data


def fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def get_align(m: Dict[str, Any]) -> Optional[float]:
    for k in ("alignment_acc", "argmax_acc"):
        if k in m and m[k] is not None:
            return float(m[k])
    return None


def get_cmce(m: Dict[str, Any]) -> Optional[float]:
    if m.get("cmce_score") is not None:
        return float(m["cmce_score"])
    return None


def get_chunk(m: Dict[str, Any]) -> Optional[float]:
    if m.get("chunk_f1") is not None:
        return float(m["chunk_f1"])
    return None


def markdown_table() -> str:
    lines = [
        "| ID | Run | alignment_acc | cmce_score | chunk_f1 | note |",
        "|----|-----|--------------:|-----------:|---------:|------|",
    ]
    for aid, folder, note in ABLATIONS:
        m = load_metrics(folder)
        src = "json" if (ROOT / "results" / folder / "metrics.json").is_file() else "fallback"
        lines.append(
            f"| {aid} | `{folder}` | {fmt(get_align(m))} | {fmt(get_cmce(m))} | "
            f"{fmt(get_chunk(m))} | {note} ({src}) |"
        )
    return "\n".join(lines) + "\n"


def latex_table() -> str:
    rows = []
    for aid, folder, note in ABLATIONS:
        m = load_metrics(folder)
        rows.append(
            f"{aid} & \\texttt{{{folder}}} & {fmt(get_align(m))} & "
            f"{fmt(get_cmce(m))} & {fmt(get_chunk(m))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{llccc}\n"
        "\\hline\n"
        "ID & Run & Align. & CMCE & Chunk F1 \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=("markdown", "latex", "both"), default="markdown")
    args = ap.parse_args()
    if args.format in ("markdown", "both"):
        print("## Ablation export (Markdown)\n")
        print(markdown_table())
    if args.format in ("latex", "both"):
        print("%% Ablation export (LaTeX)")
        print(latex_table())


if __name__ == "__main__":
    main()
