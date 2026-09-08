# tiny_real_v2

Improved real grounding prepare + short train.

## Regenerate data (images not committed)

```bash
python scripts/prepare_real_cmce.py --n-total 80 --seed 42
# optional: --use-clip if open_clip installs quickly
```

## Train

```bash
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/real/train.json --val-data data/real/val.json \
  --token-align-weight 1.25 --visual-boundary-weight 0.2 \
  --out results/tiny_real_v2
```

## vs real_v1

See `metrics.json` (`delta_vs_real_v1`). Alignment roughly flat (~0.34); structured
pseudo-GT is the main data-quality win.
