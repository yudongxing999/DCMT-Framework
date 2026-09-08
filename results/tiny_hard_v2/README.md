# tiny_hard_v2

Curriculum + longer hard train + higher token-align weight (+ optional visual boundary).

## Regenerate hard JSON

```bash
python scripts/make_synthetic_cmce.py --seed 42 --difficulty hard --n-train 300 --n-val 60
python scripts/make_synthetic_cmce.py --seed 42 --difficulty easy --n-train 200 --n-val 40
```

## Train

```bash
python scripts/train_tiny.py --epochs 15 --device cpu --seed 42 \
  --train-data data/synthetic_hard_train.json \
  --val-data data/synthetic_hard_val.json \
  --curriculum-easy-data data/synthetic_train.json \
  --curriculum-epochs 3 \
  --token-align-weight 1.75 \
  --visual-boundary-weight 0.25 \
  --out results/tiny_hard_v2
```

# Or 50/50 mix for all epochs:
#   --curriculum-mix --curriculum-easy-data data/synthetic_train.json

Hard generator keeps `alignment_token_pairs` as clean primary pairs; distractors only in `alignments`.
