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

| Run | alignment_acc | cmce_score | notes |
|-----|---------------|------------|-------|
| tiny_align_v2 (easy) | ≈0.817 | ≈0.890 | easy synthetic token-align |
| tiny_hard_v1 | ≈0.142 | ≈0.483 | hard, 8ep, no curriculum |
| tiny_hard_v2 | ≈0.356 | ≈0.607 | curriculum+15ep+w_ta=1.75 (**≥0.35 met**; stretch 0.5 missed) |
| tiny_hard_v3 | ≈0.380 | ≈0.612 | easy4→mix20, w_ta=2.25, temp=0.05, HN; **≥0.5 missed** |
| tiny_real_v1 | ≈0.344 | ≈0.606 | pseudo visual stride |
| tiny_real_v2 | ≈0.344 | ≈0.598 | NP+LTR priors; align flat |
| tiny_real_v3 | ≈0.312 | ≈0.579 | CLIP ViT-B/32 on picsum smoke |
| tiny_real_v4 | ≈0.320 | ≈0.557 | **matched Flickr8k** + CLIP; beats v3 align |

### Harder synthetic (`tiny_hard_v3`) — stretch toward 0.5

```bash
python scripts/train_tiny.py --epochs 24 --device cpu --seed 43 \
  --train-data data/synthetic_hard_train.json --val-data data/synthetic_hard_val.json \
  --curriculum-easy-data data/synthetic_train.json --curriculum-epochs 4 --curriculum-then-mix \
  --token-align-weight 2.25 --token-align-temperature 0.05 \
  --visual-boundary-weight 0.3 --hard-neg-weight 0.4 --hard-neg-boost 0.75 \
  --out results/tiny_hard_v3
```

New flags: `--curriculum-then-mix`, `--hard-neg-weight`, `--hard-neg-boost`, `--d-model`.
Best ≈0.380 — still short of 0.5 on tiny CPU smoke.

### Real CLIP grounding (`tiny_real_v3`)

```bash
pip install transformers  # CLIP (weights cache under /workspace/hf_cache)
python scripts/prepare_real_cmce.py --n-total 80 --seed 42 --grounding clip
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/real/train.json --val-data data/real/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v3
```

`--grounding {spatial,clip,auto}` (default auto). CLIP: `transformers` `openai/clip-vit-base-patch32`
(4×4 cell crops ↔ chunk text; `v = patch_idx+1`). Fallback records `grounding: spatial_fallback`.

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

`--source {auto,flickr,coco,picsum}`. Uses matched Flickr8k captions (`jxie/flickr8k`), not random picsum.
Images under `data/real_matched/images/` are regenerated (not committed). See `data/real_matched/meta.json`
and `results/tiny_real_v4/`. alignment_acc≈0.320 beats real_v3 (0.312); stretch vs real_v2 (0.344) missed.

See full package docs / prior experiment sections in git history for baseline, align_v2, hard_v1/v2, real_v1/v2 details.

## License

MIT — see `LICENSE`.

## Contact

Dongxing Yu — yudongxing@sandau.edu.cn
