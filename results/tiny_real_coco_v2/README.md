# tiny_real_coco_v2 — COCO scaled (tiny)

Scale follow-up to `tiny_real_coco_v1` (**small** model, n=100, ep=8, align≈0.254):
this run keeps **tiny** (~1.27M) with more data + epochs.

| field | value |
|-------|-------|
| model_size | tiny |
| num_params | 1,268,493 |
| n_train / n_val | 320 / 80 (n_total=400) |
| epochs | 20 |
| device / seed | cpu / 42 |
| source | coco + CLIP grounding |
| alignment_acc | 0.2908 |
| cmce_score | 0.5347 |
| chunk_f1 | 0.9004 |

## vs tiny_real_coco_v1 (small, n=100, ep=8, align≈0.254, cmce≈0.502)

- alignment_acc: 0.254 → 0.291 (↑)
- cmce_score: 0.502 → 0.535 (↑)
- Fairer tiny-vs-small comparison: scaled **tiny** on COCO beats prior **small** smoke on alignment.

## Reproduce

```bash
python scripts/prepare_real_cmce.py --source coco --out-dir data/real_coco_v2 \
  --n-total 400 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size tiny --epochs 20 --device cpu --seed 42 \
  --train-data data/real_coco_v2/train.json --val-data data/real_coco_v2/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_coco_v2
```

Images under `data/real_coco_v2/images/` are regenerated (not committed). Checkpoint `model.pt` not committed.
