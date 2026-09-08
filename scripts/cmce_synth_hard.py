"""Hard-mode helpers for synthetic CMCE generation."""
from __future__ import annotations
import random
from typing import List, Optional

OBJECTS_HARD_EXTRA = [
    "crimson automobile", "scarlet vehicle", "azure bicycle", "cyan cycle",
    "high tower", "lofty skyscraper", "verdant oak", "leafy plant",
    "tiny puppy", "little canine", "pale kitten", "ivory feline",
    "golden coach", "amber van", "dark crow", "ebony raven",
    "ancient cottage", "aged cabin", "fresh overpass", "modern span",
    "tan pony", "chestnut mare", "rose blossom", "magenta bloom",
    "ashy mist", "slate fog", "amber sphere", "copper orb",
    "metal ship", "steel vessel",
]
ACTIONS_HARD_EXTRA = [
    "is resting nearby", "appears quite still", "glides past quickly",
    "remains almost hidden", "seems rather distant", "looks slightly blurred",
]
SCENES_HARD_EXTRA = [
    "beside the crowded market stall", "across the narrow wooden bridge",
    "under the cloudy evening sky", "near the quiet harbor dock",
    "along the dusty country path", "behind the tall iron fence",
]

def apply_boundary_noise(targets, n_tokens, rng, flip_p=0.08, drop_p=0.12):
    out = list(targets)
    secondary = [i for i in range(1, min(n_tokens, len(out))) if out[i] == 1]
    if secondary and rng.random() < drop_p:
        out[rng.choice(secondary)] = 0
    candidates = [i for i in range(1, min(n_tokens, len(out))) if out[i] == 0]
    if candidates and rng.random() < flip_p:
        out[rng.choice(candidates)] = 1
    if n_tokens > 0:
        out[0] = 1
    return out

def visual_token_index_hard(visual_chunk_id, n_visual, num_patches, rng):
    base_stride = max(1, num_patches // max(n_visual, 1))
    jitter = rng.randint(-1, 2)
    stride = max(1, base_stride + jitter)
    offset = rng.randint(0, max(1, base_stride // 2 + 1))
    v = 1 + offset + visual_chunk_id * stride
    v = ((v - 1) % num_patches) + 1
    return int(min(max(v, 1), num_patches))
