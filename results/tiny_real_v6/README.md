# tiny_real_v6 — Flickr scaled (tiny)

Scale follow-up to `tiny_real_v4`: same **tiny** model, more data + epochs.

| field | value |
|-------|-------|
| model_size | tiny |
| num_params | 1,268,493 |
| n_train / n_val | 320 / 80 (n_total=400) |
| epochs | 20 |
| device / seed | cpu / 42 |
| source | flickr8k + CLIP grounding |
| alignment_acc | 0.2725 |
| cmce_score | 0.5259 |
| chunk_f1 | 0.9061 |

## vs tiny_real_v4 (n=100, ep=8, align≈0.320, cmce≈0.557)

- alignment_acc: 0.320 → 0.273 (↓)
- cmce_score: 0.557 → 0.526 (↓)
- Scaling **did not** beat v4 alignment on this CPU smoke (harder/larger val set n_val=80 vs 20 may also dilute scores).

## Reproduce

```bash
python scripts/prepare_real_cmce.py --source flickr --out-dir data/real_matched_v6 \
  --n-total 400 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size tiny --epochs 20 --device cpu --seed 42 \
  --train-data data/real_matched_v6/train.json --val-data data/real_matched_v6/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v6
```

Images under `data/real_matched_v6/images/` are regenerated (not committed). Checkpoint `model.pt` not committed.
