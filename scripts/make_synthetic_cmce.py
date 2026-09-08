#!/usr/bin/env python3
"""Generate synthetic CMCE JSON. Supports --difficulty easy|hard."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OBJECTS = ["red car","blue bike","tall building","green tree","small dog","white cat","yellow bus","black bird","old house","new bridge","brown horse","pink flower","gray cloud","orange ball","silver boat"]
OBJECTS_HARD = OBJECTS + ["crimson automobile","scarlet vehicle","azure bicycle","cyan cycle","high tower","lofty skyscraper","verdant oak","leafy plant","tiny puppy","little canine","pale kitten","ivory feline","golden coach","amber van","dark crow","ebony raven","ancient cottage","aged cabin","fresh overpass","modern span","tan pony","chestnut mare","rose blossom","magenta bloom","ashy mist","slate fog","amber sphere","copper orb","metal ship","steel vessel"]
ACTIONS = ["is parked","is running","is standing","is flying","is waiting","looks bright","seems quiet","moves slowly"]
ACTIONS_HARD = ACTIONS + ["is resting nearby","appears quite still","glides past quickly","remains almost hidden","seems rather distant","looks slightly blurred"]
SCENES = ["near the park","by the river","under sunny sky","on the street","in the garden","beside the road","across the field","next to the shop"]
SCENES_HARD = SCENES + ["beside the crowded market stall","across the narrow wooden bridge","under the cloudy evening sky","near the quiet harbor dock","along the dusty country path","behind the tall iron fence"]

def whitespace_tokens(text: str) -> List[str]:
    return text.lower().split()

def char_span_for_token_range(text: str, tok_start: int, tok_end: int) -> Tuple[int, int]:
    raw_tokens = text.split()
    if not raw_tokens:
        return 0, 0
    offsets, pos = [], 0
    for tok in raw_tokens:
        idx = text.find(tok, pos)
        if idx < 0: idx = pos
        offsets.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return offsets[tok_start][0], offsets[min(tok_end, len(offsets)) - 1][1]

def make_boundary_targets(n_tokens: int, chunk_starts: List[int], max_seq: int) -> List[int]:
    targets = [0] * max_seq
    for s in chunk_starts:
        if 0 <= s < max_seq and s < n_tokens:
            targets[s] = 1
    if n_tokens > 0:
        targets[0] = 1
    return targets

def apply_boundary_noise(targets: List[int], n_tokens: int, rng: random.Random) -> List[int]:
    out = list(targets)
    secondary = [i for i in range(1, min(n_tokens, len(out))) if out[i] == 1]
    if secondary and rng.random() < 0.12:
        out[rng.choice(secondary)] = 0
    candidates = [i for i in range(1, min(n_tokens, len(out))) if out[i] == 0]
    if candidates and rng.random() < 0.08:
        out[rng.choice(candidates)] = 1
    if n_tokens > 0:
        out[0] = 1
    return out

def visual_token_index(vid: int, n_visual: int, num_patches: int, rng: Optional[random.Random]=None, hard: bool=False) -> int:
    # Hard: mild LTR column jitter so primary pairs stay learnable; distractors stay in alignments only.
    if hard and rng is not None:
        base = max(1, num_patches // max(n_visual, 1))
        stride = max(1, base + rng.randint(0, 1))
        offset = rng.randint(0, max(0, base // 3))
        v = 1 + offset + vid * stride
        if v > num_patches:
            v = ((v - 1) % num_patches) + 1
        return int(min(max(v, 1), num_patches))
    stride = max(1, num_patches // max(n_visual, 1))
    return int(min(max(1 + vid * stride, 1), num_patches))

def _pick(pool: List[str], used: set, rng: random.Random) -> str:
    phrase, tries = rng.choice(pool), 0
    while phrase in used and tries < 20:
        phrase = rng.choice(pool); tries += 1
    used.add(phrase)
    return phrase

def make_sample(idx: int, rng: random.Random, max_seq: int, num_labels: int, img_size: int=64, patch_size: int=16, difficulty: str="easy") -> Dict[str, Any]:
    hard = difficulty == "hard"
    objects = OBJECTS_HARD if hard else OBJECTS
    actions = ACTIONS_HARD if hard else ACTIONS
    scenes = SCENES_HARD if hard else SCENES
    pools = [objects, actions, scenes]
    n_chunks = rng.randint(3, 5) if hard else rng.randint(2, 3)
    phrases, used = [], set()
    for i in range(n_chunks):
        phrases.append(_pick(pools[i % len(pools)], used, rng))
    if hard and rng.random() < 0.35 and len(phrases) >= 2:
        stem = phrases[0].split()[-1]
        conf = [o for o in objects if stem in o.split() or o.split()[-1][:3] == stem[:3]]
        if conf:
            alt = rng.choice(conf)
            if alt != phrases[0]:
                phrases[rng.randint(1, len(phrases)-1)] = alt
    if hard and rng.random() < 0.4:
        extra = rng.choice(scenes)
        if extra not in phrases:
            phrases.append(extra); n_chunks = len(phrases)
    text = " ".join(phrases)
    tokens = whitespace_tokens(text)
    if len(tokens) > max_seq:
        tokens = tokens[:max_seq]; text = " ".join(text.split()[:max_seq])
    chunk_starts, text_chunks, cursor = [], [], 0
    for cid, phrase in enumerate(phrases):
        ptoks = whitespace_tokens(phrase)
        start_tok, end_tok = cursor, cursor + len(ptoks)
        if start_tok >= max_seq: break
        end_tok = min(end_tok, max_seq)
        chunk_starts.append(start_tok)
        cs, ce = char_span_for_token_range(text, start_tok, end_tok)
        text_chunks.append({"id": cid, "start": cs, "end": ce, "token_start": start_tok, "token_end": end_tok, "text": " ".join(text.split()[start_tok:end_tok])})
        cursor = end_tok
    n_visual = len(text_chunks)
    visual_chunks = []
    for vid in range(n_visual):
        x0, y0 = 10 + vid * 40, 10 + (vid % 2) * 30
        lab = phrases[vid].split()[-1] if vid < len(phrases) else f"obj_{vid}"
        attrs = phrases[vid].split()[:-1] if vid < len(phrases) else []
        visual_chunks.append({"id": vid, "bbox": [x0, y0, x0+80, y0+80], "label": lab, "attributes": attrs})
    num_patches = (img_size // patch_size) ** 2
    alignments, alignment_token_pairs, used_v = [], [], set()
    for vid in range(n_visual):
        tid = min(vid, len(text_chunks) - 1)
        alignments.append({"visual_chunk": vid, "text_chunk": tid, "confidence": round(rng.uniform(0.75, 0.98), 2), "primary": True})
        v_tok = visual_token_index(vid, n_visual, num_patches, rng=rng, hard=hard)
        if v_tok in used_v:
            for cand in range(1, num_patches + 1):
                if cand not in used_v:
                    v_tok = cand; break
        used_v.add(v_tok)
        t_tok = chunk_starts[tid] if tid < len(chunk_starts) else 0
        alignment_token_pairs.append({"v": int(v_tok), "t": int(t_tok)})
        if len(text_chunks) > 1 and rng.random() < (0.55 if hard else 0.3):
            other = (tid + 1) % len(text_chunks)
            alignments.append({"visual_chunk": vid, "text_chunk": other, "confidence": round(rng.uniform(0.55, 0.75), 2), "primary": False})
        if hard and len(text_chunks) > 2 and rng.random() < 0.4:
            wrong = rng.choice([c for c in range(len(text_chunks)) if c != tid])
            alignments.append({"visual_chunk": vid, "text_chunk": wrong, "confidence": round(rng.uniform(0.25, 0.55), 2), "primary": False, "distractor": True})
    label = int(hashlib.md5(((phrases[0] if phrases else "") + f"|{idx}|{difficulty}").encode()).hexdigest(), 16) % num_labels
    targets = make_boundary_targets(len(tokens), chunk_starts, max_seq)
    if hard:
        targets = apply_boundary_noise(targets, len(tokens), rng)
    return {
        "id": f"synth_{difficulty}_{idx:05d}" if hard else f"synth_{idx:05d}",
        "image_path": f"images/synth_{idx:05d}.jpg",
        "text": text, "label": label, "text_boundary_targets": targets,
        "visual_chunks": visual_chunks, "text_chunks": text_chunks,
        "alignments": alignments, "alignment_token_pairs": alignment_token_pairs,
        "metadata": {"complexity": n_chunks, "num_objects": n_visual, "text_length": len(text), "n_tokens": len(tokens), "num_patches": num_patches, "img_size": img_size, "patch_size": patch_size, "source": "synthetic", "difficulty": difficulty},
    }

def main():
    p = argparse.ArgumentParser(description="Generate synthetic CMCE JSON splits")
    p.add_argument("--out-dir", default="data"); p.add_argument("--n-train", type=int, default=200); p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--max-seq", type=int, default=64); p.add_argument("--num-labels", type=int, default=10)
    p.add_argument("--img-size", type=int, default=64); p.add_argument("--patch-size", type=int, default=16); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", choices=["easy", "hard"], default="easy")
    p.add_argument("--out-prefix", default=None, help="default: synthetic or synthetic_hard")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or ("synthetic_hard" if args.difficulty == "hard" else "synthetic")
    rng = random.Random(args.seed)
    train = [make_sample(i, rng, args.max_seq, args.num_labels, args.img_size, args.patch_size, args.difficulty) for i in range(args.n_train)]
    val = [make_sample(args.n_train + i, rng, args.max_seq, args.num_labels, args.img_size, args.patch_size, args.difficulty) for i in range(args.n_val)]
    (out / f"{prefix}_train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (out / f"{prefix}_val.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    print(f"difficulty={args.difficulty} prefix={prefix} train={len(train)} val={len(val)}")
    print("example:", train[0]["text"], train[0]["alignment_token_pairs"])

if __name__ == "__main__":
    main()
