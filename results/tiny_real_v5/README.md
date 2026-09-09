# tiny_real_v5 — Flickr matched + CLIP + **small** model

CPU smoke run on the same Flickr8k matched split as `tiny_real_v4`, but with
`--model-size small` (wider/deeper than the prior `tiny` preset).

## Config

| field | tiny (v4) | **small (v5)** |
|-------|-----------|----------------|
| d_model | 64 | **128** |
| n_heads | 4 | 4 |
| n_layers | 1 | **2** |
| d_ff | 128 | **256** |
| v_layers | 1 | **2** |
| img_size / patch | 64 / 16 | 64 / 16 |
| vocab / max_seq / labels | 1000 / 64 / 10 | same |
| dropout | 0.0 | **0.05** |
| params | ~1.27M | **~3.19M** |

## Command

```bash
python scripts/prepare_real_cmce.py --source flickr --out-dir data/real_matched \
  --n-total 100 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size small --epochs 8 --device cpu --seed 42 \
  --train-data data/real_matched/train.json --val-data data/real_matched/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v5
```

## Results (seed=42)

| metric | tiny_real_v4 (tiny) | **tiny_real_v5 (small)** |
|--------|---------------------|--------------------------|
| alignment_acc | ≈0.320 | **≈0.156** |
| cmce_score | ≈0.557 | **≈0.471** |
| chunk_f1 | ≈0.912 | **≈0.945** |
| params | 1,268,493 | 3,194,893 |

**Takeaway:** on this tiny CPU / n=100 smoke set, the larger `small` preset did
**not** help token alignment (dropped ~0.16 abs). Chunk F1 rose slightly.
Likely under-trained / over-parameterized for 80 train samples × 8 epochs on CPU.
Do **not** commit `model.pt` or `data/real_matched/images/`.

See repo root README comparison table.
