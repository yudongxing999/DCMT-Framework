# tiny_real_v4 — matched Flickr8k + CLIP

Replaces picsum smoke images with **real matched** Flickr8k image–caption pairs
(`jxie/flickr8k`), CLIP patch-grid grounding, same tiny train recipe as real_v3.

## Result

| metric | real_v2 (picsum+spatial) | real_v3 (picsum+CLIP) | **real_v4 (flickr+CLIP)** |
|--------|--------------------------|------------------------|---------------------------|
| alignment_acc | 0.3438 | 0.3125 | **0.3200** |
| cmce_score | 0.5979 | 0.5792 | **0.5570** |
| chunk_f1 | 0.9792 | 0.9792 | 0.9124 |

- Beats real_v3 on `alignment_acc` (+0.0075); stretch goal vs real_v2 missed (−0.0238).
- `cmce_score` dips because chunk_f1 is lower on diverse Flickr phrasing.
- Attempt1 (unfiltered captions, max 27 tokens): alignment_acc≈0.119 — too hard for tiny smoke.
- Curriculum ablation (easy→matched): alignment_acc≈0.270 — did not help.

## Data

- Source: `jxie/flickr8k` (HF parquet; matched captions)
- n=100 (train 80 / val 20), `--max-caption-tokens 12`
- Images under `data/real_matched/images/` — **not committed**; regenerate
- Spot-check (caption describes image):
  - `flickr8k_05691`: "A pitbull dog is biting another dog on the face ."
  - `flickr8k_00426`: "A dog runs by the water 's edge ."
  - `flickr8k_01098`: "A girl rides on a swing ."

## Reproduce

```bash
pip install datasets transformers Pillow
python scripts/prepare_real_cmce.py --source flickr --out-dir data/real_matched \
  --n-total 100 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/real_matched/train.json --val-data data/real_matched/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v4
```

CLIP: `transformers:openai/clip-vit-base-patch32` (cache `/workspace/hf_cache`).
