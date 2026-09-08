#!/usr/bin/env python3
"""Generate tiny synthetic CMCE train/val JSON for baseline training.

Outputs:
  data/synthetic_train.json  (200 samples)
  data/synthetic_val.json    (40 samples)

Schema matches evaluate.load_cmce_json plus training fields:
  - label: int class id in [0, num_labels)
  - text_boundary_targets: list[int] length=max_seq (0/1), aligned to
    evaluate.simple_tokenize whitespace tokens
  - alignment_token_pairs: list of {"v": int, "t": int} for primary
    (high-confidence) visual↔text alignments used for token-level supervision
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Phrase bank: each phrase is a short noun/verb phrase used as a text chunk.
OBJECTS = [
    "red car",
    "blue bike",
    "tall building",
    "green tree",
    "small dog",
    "white cat",
    "yellow bus",
    "black bird",
    "old house",
    "new bridge",
    "brown horse",
    "pink flower",
    "gray cloud",
    "orange ball",
    "silver boat",
]

ACTIONS = [
    "is parked",
    "is running",
    "is standing",
    "is flying",
    "is waiting",
    "looks bright",
    "seems quiet",
    "moves slowly",
]

SCENES = [
    "near the park",
    "by the river",
    "under sunny sky",
    "on the street",
    "in the garden",
    "beside the road",
    "across the field",
    "next to the shop",
]


def whitespace_tokens(text: str) -> List[str]:
    return text.lower().split()


def char_span_for_token_range(text: str, tok_start: int, tok_end: int) -> Tuple[int, int]:
    """Map inclusive token index range [tok_start, tok_end) to char spans in original text."""
    raw_tokens = text.split()
    if not raw_tokens:
        return 0, 0
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for tok in raw_tokens:
        idx = text.find(tok, pos)
        if idx < 0:
            idx = pos
        offsets.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    s = offsets[tok_start][0]
    e = offsets[min(tok_end, len(offsets)) - 1][1]
    return s, e


def make_boundary_targets(n_tokens: int, chunk_starts: List[int], max_seq: int) -> List[int]:
    targets = [0] * max_seq
    for s in chunk_starts:
        if 0 <= s < max_seq and s < n_tokens:
            targets[s] = 1
    if n_tokens > 0 and targets[0] == 0:
        targets[0] = 1
    return targets


def visual_token_index(
    visual_chunk_id: int,
    n_visual: int,
    num_patches: int,
) -> int:
    """Map visual chunk id -> patch token index in [1, num_patches] (skip CLS=0)."""
    stride = max(1, num_patches // max(n_visual, 1))
    v = 1 + visual_chunk_id * stride
    return int(min(max(v, 1), num_patches))


def make_sample(
    idx: int,
    rng: random.Random,
    max_seq: int,
    num_labels: int,
    img_size: int = 64,
    patch_size: int = 16,
) -> Dict[str, Any]:
    n_chunks = rng.randint(2, 3)
    phrases: List[str] = []
    pools = [OBJECTS, ACTIONS, SCENES]
    used = set()
    for i in range(n_chunks):
        pool = pools[i % len(pools)]
        phrase = rng.choice(pool)
        tries = 0
        while phrase in used and tries < 10:
            phrase = rng.choice(pool)
            tries += 1
        used.add(phrase)
        phrases.append(phrase)

    text = " ".join(phrases)
    tokens = whitespace_tokens(text)
    if len(tokens) > max_seq:
        tokens = tokens[:max_seq]
        text = " ".join(text.split()[:max_seq])

    chunk_starts: List[int] = []
    cursor = 0
    text_chunks = []
    for cid, phrase in enumerate(phrases):
        ptoks = whitespace_tokens(phrase)
        start_tok = cursor
        end_tok = cursor + len(ptoks)
        if start_tok >= max_seq:
            break
        end_tok = min(end_tok, max_seq)
        chunk_starts.append(start_tok)
        c_start, c_end = char_span_for_token_range(text, start_tok, end_tok)
        text_chunks.append(
            {
                "id": cid,
                "start": c_start,
                "end": c_end,
                "token_start": start_tok,
                "token_end": end_tok,
                "text": " ".join(text.split()[start_tok:end_tok]),
            }
        )
        cursor = end_tok

    n_visual = n_chunks
    visual_chunks = []
    for vid in range(n_visual):
        x0 = 10 + vid * 40
        y0 = 10 + (vid % 2) * 30
        visual_chunks.append(
            {
                "id": vid,
                "bbox": [x0, y0, x0 + 80, y0 + 80],
                "label": phrases[vid].split()[-1],
                "attributes": phrases[vid].split()[:-1],
            }
        )

    num_patches = (img_size // patch_size) ** 2

    # Primary high-confidence 1:1 alignments + clean token pairs for supervision.
    # Secondary noisy pairs (p=0.3) are omitted from alignment_token_pairs.
    alignments = []
    alignment_token_pairs: List[Dict[str, int]] = []
    for vid in range(n_visual):
        tid = min(vid, len(text_chunks) - 1)
        alignments.append(
            {
                "visual_chunk": vid,
                "text_chunk": tid,
                "confidence": round(rng.uniform(0.75, 0.98), 2),
                "primary": True,
            }
        )
        v_tok = visual_token_index(vid, n_visual, num_patches)
        t_tok = chunk_starts[tid] if tid < len(chunk_starts) else 0
        alignment_token_pairs.append({"v": int(v_tok), "t": int(t_tok)})

        # Keep secondary noise in annotations only (not in token-pair supervision).
        if len(text_chunks) > 1 and rng.random() < 0.3:
            other = (tid + 1) % len(text_chunks)
            alignments.append(
                {
                    "visual_chunk": vid,
                    "text_chunk": other,
                    "confidence": round(rng.uniform(0.55, 0.75), 2),
                    "primary": False,
                }
            )

    key = (phrases[0] if phrases else "") + f"|{idx}"
    digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
    label = digest % num_labels

    targets = make_boundary_targets(len(tokens), chunk_starts, max_seq)

    return {
        "id": f"synth_{idx:05d}",
        "image_path": f"images/synth_{idx:05d}.jpg",
        "text": text,
        "label": int(label),
        "text_boundary_targets": targets,
        "visual_chunks": visual_chunks,
        "text_chunks": text_chunks,
        "alignments": alignments,
        "alignment_token_pairs": alignment_token_pairs,
        "metadata": {
            "complexity": n_chunks,
            "num_objects": n_visual,
            "text_length": len(text),
            "n_tokens": len(tokens),
            "num_patches": num_patches,
            "img_size": img_size,
            "patch_size": patch_size,
            "source": "synthetic",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CMCE JSON splits")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--max-seq", type=int, default=64)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train = [
        make_sample(
            i, rng, args.max_seq, args.num_labels, args.img_size, args.patch_size
        )
        for i in range(args.n_train)
    ]
    val = [
        make_sample(
            args.n_train + i,
            rng,
            args.max_seq,
            args.num_labels,
            args.img_size,
            args.patch_size,
        )
        for i in range(args.n_val)
    ]

    train_path = out_dir / "synthetic_train.json"
    val_path = out_dir / "synthetic_val.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)

    print(f"Wrote {len(train)} train samples -> {train_path}")
    print(f"Wrote {len(val)} val samples -> {val_path}")
    print(f"Example text: {train[0]['text']}")
    print(f"Example label: {train[0]['label']}")
    print(f"Example alignment_token_pairs: {train[0]['alignment_token_pairs']}")
    n_ones = sum(train[0]["text_boundary_targets"])
    print(f"Example boundary 1-count: {n_ones}/{args.max_seq}")


if __name__ == "__main__":
    main()
