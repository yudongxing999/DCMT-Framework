# Hard synthetic JSON

Regenerate (preferred over committing multi‑MB JSON):

```bash
python scripts/make_synthetic_cmce.py --seed 42 --difficulty hard --n-train 300 --n-val 60
```

Writes `data/synthetic_hard_train.json` / `data/synthetic_hard_val.json`.

Then train:

```bash
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/synthetic_hard_train.json --val-data data/synthetic_hard_val.json \
  --out results/tiny_hard_v1
```
