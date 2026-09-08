# tiny_hard_v3

Goal: push hard synthetic `alignment_acc` from hard_v2 ≈0.356 toward **≥0.5**.

## Result

| metric | hard_v2 | hard_v3 (best) |
|--------|---------|----------------|
| alignment_acc | 0.3561 | **0.3803** |
| cmce_score | 0.6067 | **0.6124** |
| chunk_f1 | 0.9827 | 0.9605 |

**Target ≥0.5: MISSED** (best ≈0.380).

## Recipe (best — seed 43)

```bash
python scripts/make_synthetic_cmce.py --seed 42 --difficulty hard --n-train 300 --n-val 60
python scripts/make_synthetic_cmce.py --seed 42 --difficulty easy --n-train 200 --n-val 40
python scripts/train_tiny.py --epochs 24 --device cpu --seed 43 \
  --train-data data/synthetic_hard_train.json \
  --val-data data/synthetic_hard_val.json \
  --curriculum-easy-data data/synthetic_train.json \
  --curriculum-epochs 4 --curriculum-then-mix \
  --token-align-weight 2.25 --token-align-temperature 0.05 \
  --visual-boundary-weight 0.3 \
  --hard-neg-weight 0.4 --hard-neg-boost 0.75 \
  --out results/tiny_hard_v3
```

## Attempts

1. seed=42 then-mix → ≈0.373
2. seed=42 hard-only + stronger HN/temp → ≈0.359 (worse)
3. seed=43 then-mix → **≈0.380** (best; checkpoint)

## Notes

- Eval uses the same `alignment_token_pairs` as supervision.
- Hard-neg: distractor text tokens from non-primary `alignments`.
- Mix after curriculum is length-capped (~50/50, no doubled epoch).
