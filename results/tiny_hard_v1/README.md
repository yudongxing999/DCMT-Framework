# tiny_hard_v1

Harder synthetic CMCE train (seed=42, epochs=8, CPU).

```bash
python scripts/make_synthetic_cmce.py --seed 42 --difficulty hard --n-train 300 --n-val 60
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/synthetic_hard_train.json --val-data data/synthetic_hard_val.json \
  --out results/tiny_hard_v1
```

Compared to easy `tiny_align_v2` (alignment_acc≈0.817, cmce≈0.890), hard scores
are intentionally lower (irregular visual strides, distractors, boundary noise).
See `metrics.json`.
