# tiny_real_coco_v1 — COCO matched + CLIP + **small** model

Same recipe as `tiny_real_v5` (Flickr small), but on a COCO-2017 caption subset
(`phiyodr/coco2017`, n=100, CLIP grounding, max-caption-tokens=12).

## Spot-check (image ↔ caption)

| id | caption |
|----|---------|
| coco_000000324927 | A plate of two slices of pizza and a cup of juice. |
| coco_000000058111 | An orange and white cat laying on top of a sink. |
| coco_000000460682 | a horse that is laying down in a field |
| coco_000000189078 | Variety of fruits such as bananas, oranges, and apples on a platter. |
| coco_000000415882 | a woman lying on a bed underneath a blanket |

Images under `data/real_coco/images/` verified present; captions are matched COCO
captions (not picsum placeholders). See `data/real_coco/meta.json`.

## Command

```bash
python scripts/prepare_real_cmce.py --source coco --out-dir data/real_coco \
  --n-total 100 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size small --epochs 8 --device cpu --seed 42 \
  --train-data data/real_coco/train.json --val-data data/real_coco/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_coco_v1
```

## Results (seed=42, model-size=small, ~3.19M params)

| metric | value |
|--------|-------|
| alignment_acc | ≈0.254 |
| cmce_score | ≈0.502 |
| chunk_f1 | ≈0.873 |

Vs Flickr small (`tiny_real_v5` align≈0.156): COCO small is **higher** on alignment.
Vs Flickr tiny (`tiny_real_v4` align≈0.320): both small runs still lag the tiny Flickr baseline.

Do **not** commit `model.pt` or `data/real_coco/images/`.
