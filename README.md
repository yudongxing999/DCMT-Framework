# DCMT Framework

**Dynamic Cross-Modal Tokenization** — adaptive token boundaries that integrate human chunking mechanisms into multimodal models.

Paper: *Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs*

## Migration note

This is the clean public package layout migrated from the oddly named source repo
[`yudonald999-sys/-yudongxing-DCMT-Framework.`](https://github.com/yudonald999-sys/-yudongxing-DCMT-Framework.)
(trailing period in the repo name). Imports, package structure, and several
runtime bugs are fixed here. Large figure PNGs from the old repo are **not**
copied (they remain available in the source repository).

## Install

```bash
git clone https://github.com/yudongxing999/DCMT-Framework.git
cd DCMT-Framework

pip install -r requirements.txt
pip install -e .
```

Requirements: `torch`, `numpy`, `scipy`, `tqdm`, `Pillow`, `torchvision`.
Optional CLIP grounding: `transformers>=4.36` (see requirements.txt).
Optional matched Flickr/COCO: `datasets` (`jxie/flickr8k`, `phiyodr/coco2017`).

## Comparison table (CPU tiny smoke)

| Run | alignment_acc | cmce_score | chunk_f1 | params | notes |
|-----|---------------|------------|----------|--------|-------|
| tiny_align_v2 (easy) | ≈0.817 | ≈0.890 | ≈1.00 | ~1.27M | easy synthetic token-align |
| tiny_hard_v1 | ≈0.142 | ≈0.483 | ≈0.99 | ~1.27M | hard, 8ep, no curriculum |
| tiny_hard_v2 | ≈0.356 | ≈0.607 | — | ~1.27M | curriculum+15ep (**≥0.35 met**) |
| tiny_hard_v3 | ≈0.380 | ≈0.612 | — | ~1.27M | stretch 0.5 missed |
| tiny_real_v1 | ≈0.344 | ≈0.606 | ≈1.00 | ~1.27M | pseudo visual stride |
| tiny_real_v2 | ≈0.344 | ≈0.598 | — | ~1.27M | NP+LTR priors |
| tiny_real_v3 | ≈0.312 | ≈0.579 | — | ~1.27M | CLIP on picsum |
| tiny_real_v4 | ≈0.320 | ≈0.557 | ≈0.912 | ~1.27M | **Flickr matched** + CLIP; tiny |
| tiny_real_v5 | ≈0.156 | ≈0.471 | ≈0.945 | **~3.19M** | Flickr + CLIP + **small**; align ↓ |
| tiny_real_coco_v1 | ≈0.254 | ≈0.502 | ≈0.873 | **~3.19M** | **COCO** matched + CLIP + small |

### Model size presets (`--model-size`)

`scripts/train_tiny.py` supports:
- `tiny` (default legacy): d_model=64, n_layers=1, d_ff=128, v_layers=1, dropout=0
- `small`: d_model=128, n_layers=2, d_ff=256, v_layers=2, dropout=0.05 (~3.19M params)

`metrics.json` records `model_size`, `num_params`, and full `config`.

### Real Flickr small (`tiny_real_v5`)

```bash
python scripts/train_tiny.py --model-size small --epochs 8 --device cpu --seed 42 \
  --train-data data/real_matched/train.json --val-data data/real_matched/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v5
```

Honest takeaway: on n=100 CPU smoke, **larger model did not help** token alignment
(0.156 vs v4 0.320); chunk_f1 rose slightly.

### Real COCO small (`tiny_real_coco_v1`)

```bash
python scripts/prepare_real_cmce.py --source coco --out-dir data/real_coco \
  --n-total 100 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size small --epochs 8 --device cpu --seed 42 \
  --train-data data/real_coco/train.json --val-data data/real_coco/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_coco_v1
```

HF: `phiyodr/coco2017`. Spot-check captions match images (pizza/juice, cat/sink, horse/field, fruit platter, person in bed).
COCO-small align≈0.254 > Flickr-small (0.156) but < Flickr-tiny v4 (0.320).

### Real matched Flickr (`tiny_real_v4`)

```bash
pip install datasets transformers
python scripts/prepare_real_cmce.py --source flickr --out-dir data/real_matched \
  --n-total 100 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/real_matched/train.json --val-data data/real_matched/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v4
```

Images under `data/*/images/` are regenerated (not committed). See `results/tiny_real_v4|v5|coco_v1/`.

## License

MIT — see `LICENSE`.

## Contact

Dongxing Yu — yudongxing@sandau.edu.cn
