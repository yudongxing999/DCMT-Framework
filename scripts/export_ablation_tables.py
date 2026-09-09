#!/usr/bin/env python3
"""Export paper-ready ablation tables (Markdown + LaTeX) from committed metrics.json.

Usage (from repo root):
  python scripts/export_ablation_tables.py
  python scripts/export_ablation_tables.py --results-dir results --format both
  python scripts/export_ablation_tables.py --format markdown
  python scripts/export_ablation_tables.py --format latex

Reads results/<folder>/metrics.json for fixed ablation IDs A0–A6.
Missing folders print N/A (does not fail). No training is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fixed ablation matrix (must stay in sync with docs/PAPER_ABLATIONS.md)
ABLATIONS: List[Dict[str, str]] = [
    {
        "id": "A0",
        "folder": "tiny_baseline",
        "setting": "tiny, no token InfoNCE (easy synth)",
        "role": "baseline",
    },
    {
        "id": "A1",
        "folder": "tiny_align_v2",
        "setting": "tiny + token InfoNCE (easy synth)",
        "role": "InfoNCE on",
    },
    {
        "id": "A2",
        "folder": "tiny_hard_v1",
        "setting": "tiny + InfoNCE, hard synth, 8ep",
        "role": "hard no curriculum",
    },
    {
        "id": "A3",
        "folder": "tiny_hard_v2",
        "setting": "tiny + InfoNCE, hard + curriculum",
        "role": "hard curriculum",
    },
    {
        "id": "A4",
        "folder": "tiny_hard_v3",
        "setting": "tiny + InfoNCE, hard stretch (seed 43)",
        "role": "hard best",
    },
    {
        "id": "A5",
        "folder": "tiny_real_v4",
        "setting": "tiny + CLIP, Flickr matched n=100 ep=8",
        "role": "best matched real",
    },
    {
        "id": "A6a",
        "folder": "tiny_real_v5",
        "setting": "small width + CLIP, Flickr n=100 ep=8",
        "role": "neg: width",
    },
    {
        "id": "A6b",
        "folder": "tiny_real_v6",
        "setting": "tiny + CLIP, Flickr scaled n=400 ep=20",
        "role": "neg: scale n",
    },
]

METRIC_KEYS = ("chunk_f1", "alignment_acc", "cmce_score")


def _pick_metric(data: Dict[str, Any], key: str) -> Optional[float]:
    """Prefer nested metrics dict, then top-level key."""
    nested = data.get("metrics")
    if isinstance(nested, dict) and key in nested and nested[key] is not None:
        try:
            return float(nested[key])
        except (TypeError, ValueError):
            pass
    if key in data and data[key] is not None:
        try:
            return float(data[key])
        except (TypeError, ValueError):
            pass
    return None


def load_metrics(results_dir: Path, folder: str) -> Dict[str, Optional[float]]:
    path = results_dir / folder / "metrics.json"
    out: Dict[str, Optional[float]] = {k: None for k in METRIC_KEYS}
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out
    for k in METRIC_KEYS:
        out[k] = _pick_metric(data, k)
    return out


def fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def collect_rows(results_dir: Path) -> List[Tuple[Dict[str, str], Dict[str, Optional[float]]]]:
    return [(abl, load_metrics(results_dir, abl["folder"])) for abl in ABLATIONS]


def render_markdown(rows: List[Tuple[Dict[str, str], Dict[str, Optional[float]]]]) -> str:
    lines = [
        "Table: Ablation matrix A0–A6 (CPU smoke; alignment_acc = argmax).",
        "",
        "| ID | Folder | Setting | chunk_f1 | align_acc | cmce_score | Role |",
        "|----|--------|---------|---------:|----------:|-----------:|------|",
    ]
    for abl, m in rows:
        lines.append(
            f"| {abl['id']} | `{abl['folder']}` | {abl['setting']} | "
            f"{fmt(m['chunk_f1'])} | {fmt(m['alignment_acc'])} | {fmt(m['cmce_score'])} | "
            f"{abl['role']} |"
        )
    lines.append("")
    lines.append(
        "*Source of truth: `results/*/metrics.json`. Regenerate with "
        "`python scripts/export_ablation_tables.py`.*"
    )
    return "\n".join(lines)


def render_latex(rows: List[Tuple[Dict[str, str], Dict[str, Optional[float]]]]) -> str:
    lines = [
        "% Ablation matrix A0--A6 (CPU smoke; alignment\\_acc = argmax).",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Ablation matrix A0--A6 on the fixed tiny protocol "
        "(CPU smoke). \\texttt{alignment\\_acc} uses argmax over "
        "token--patch scores. Not official CMCE.}",
        "  \\label{tab:dcmt-ablations}",
        "  \\begin{tabular}{llp{4.2cm}rrr}",
        "    \\toprule",
        "    ID & Folder & Setting & chunk\\_f1 & align\\_acc & cmce\\_score \\\\",
        "    \\midrule",
    ]
    for abl, m in rows:
        setting = abl["setting"].replace("_", "\\_")
        folder = abl["folder"].replace("_", "\\_")
        lines.append(
            f"    {abl['id']} & \\texttt{{{folder}}} & {setting} & "
            f"{fmt(m['chunk_f1'])} & {fmt(m['alignment_acc'])} & "
            f"{fmt(m['cmce_score'])} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Markdown/LaTeX ablation tables from results/*/metrics.json"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing run folders (default: results)",
    )
    parser.add_argument(
        "--format",
        choices=("both", "markdown", "latex"),
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args()
    rows = collect_rows(args.results_dir)

    parts: List[str] = []
    if args.format in ("both", "markdown"):
        parts.append("=== Markdown ===")
        parts.append(render_markdown(rows))
    if args.format in ("both", "latex"):
        if parts:
            parts.append("")
        parts.append("=== LaTeX ===")
        parts.append(render_latex(rows))
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
