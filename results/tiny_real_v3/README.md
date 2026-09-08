# tiny_real_v3

CLIP patch-grid grounding for real image-text CMCE (tiny / smoke).

## Result

| metric | real_v2 | real_v3 |
|--------|---------|---------|
| alignment_acc | 0.3438 | **0.3125** |
| cmce_score | 0.5979 | **0.5792** |
| chunk_f1 | 0.9792 | 0.9792 |

## CLIP

- Model: `transformers:openai/clip-vit-base-patch32`
- Grounding: `clip`
- Cache: `/workspace/hf_cache` (transformers) — see `data/real/meta.json`

## Reproduce

```bash
pip install transformers  # CLIP
python scripts/prepare_real_cmce.py --n-total 80 --seed 42 --grounding clip
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/real/train.json --val-data data/real/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v3
```

## Notes

Pseudo-GT: each text chunk’s CLIP text embed is matched to a 4×4 cell crop;
best patch index +1 (CLS offset) → `alignment_token_pairs.v`. Picsum noise
images often keep LTR spatial prior; curated photos can deviate. Flat/slightly
lower align vs real_v2 is expected on this tiny CPU smoke set.
