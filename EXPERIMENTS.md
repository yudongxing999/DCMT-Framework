# DCMT / CMCE Experiment Log (Reproducible)

Status: **paused** at this document. Numbers below are CPU smoke runs for pipeline validation, **not** paper-ready CMCE benchmarks.

Repo: https://github.com/yudongxing999/DCMT-Framework  
Archived source (figures): https://github.com/yudonald999-sys/-yudongxing-DCMT-Framework.

## Protocol (common)

| Item | Value |
|------|-------|
| Device | CPU |
| Seed | 42 (hard_v3 best used seed 43) |
| Metrics | `chunk_f1`, `alignment_acc` (=argmax), `threshold_acc`, `cmce_score` |
| Tiny preset | d_model=64, n_heads=4, n_layers=1, d_ff=128, v_layers=1, img_size=64, patch=16 (~1.27M) |
| Small preset | d_model=128, n_layers=2, d_ff=256, v_layers=2 (~3.19M) |
| Loss (typical) | CE + contrastive + boundary + token-align InfoNCE |
| Artifacts | `results/*/metrics.json` committed; images/`model.pt` **not** committed |

Install:

```bash
git clone https://github.com/yudongxing999/DCMT-Framework.git
cd DCMT-Framework
pip install -r requirements.txt && pip install -e .
# optional: transformers datasets  (CLIP + Flickr/COCO)
```

Smoke (no data download):

```bash
python -m pytest tests/test_forward_smoke.py -q
```

---

## 1. Synthetic CMCE

### 1.1 Easy + token-align (`tiny_align_v2`)

| Metric | Score |
|--------|------:|
| alignment_acc | ≈0.817 |
| cmce_score | ≈0.890 |
| chunk_f1 | ≈1.00 |

```bash
python scripts/make_synthetic_cmce.py --seed 42 --difficulty easy
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 --out results/tiny_align_v2
```

Earlier baseline without token InfoNCE (`tiny_baseline`): align≈0.288 → InfoNCE lift +0.53.

### 1.2 Hard difficulty

| Run | align | cmce | Notes |
|-----|------:|-----:|-------|
| tiny_hard_v1 | ≈0.142 | ≈0.483 | 8ep, no curriculum |
| tiny_hard_v2 | ≈0.356 | ≈0.607 | curriculum; ≥0.35 target met |
| tiny_hard_v3 | ≈0.380 | ≈0.612 | stretch ≥0.5 **missed** |

```bash
python scripts/make_synthetic_cmce.py --seed 42 --difficulty hard --n-train 300 --n-val 60
# hard_v2-style curriculum (see scripts/train_tiny.py flags)
python scripts/train_tiny.py --epochs 15 --device cpu --seed 42 \
  --train-data data/synthetic_hard_train.json --val-data data/synthetic_hard_val.json \
  --curriculum-easy-data data/synthetic_train.json --curriculum-epochs 3 \
  --out results/tiny_hard_v2
```

Hard drops are expected (distractors, irregular visual map, boundary noise).

---

## 2. Real image–text (matched)

Pseudo / CLIP visual GT is **weak supervision**, not human CMCE labels.

### 2.1 Evolution on real data

| Run | Data | Grounding | Model | n | ep | align | cmce |
|-----|------|-----------|-------|---|---:|------:|-----:|
| tiny_real_v1 | picsum | stride pseudo | tiny | ~80 | 5 | ≈0.344 | ≈0.606 |
| tiny_real_v2 | picsum | NP+LTR spatial | tiny | ~80 | 5 | ≈0.344 | ≈0.598 |
| tiny_real_v3 | picsum | CLIP | tiny | ~80 | 8 | ≈0.312 | ≈0.579 |
| **tiny_real_v4** | **Flickr8k** | CLIP | tiny | 80/20 | 8 | **≈0.320** | **≈0.557** |
| tiny_real_v5 | Flickr8k | CLIP | **small** | 80/20 | 8 | ≈0.156 | ≈0.471 |
| tiny_real_v6 | Flickr8k | CLIP | tiny | 320/80 | 20 | ≈0.273 | ≈0.526 |
| tiny_real_coco_v1 | COCO | CLIP | **small** | 80/20 | 8 | ≈0.254 | ≈0.502 |
| tiny_real_coco_v2 | COCO | CLIP | tiny | 320/80 | 20 | ≈0.291 | ≈0.535 |

Best **matched** tiny alignment so far: **Flickr v4 (≈0.320)**.

### 2.2 Reproduce Flickr v4 (recommended real checkpoint)

```bash
pip install datasets transformers
python scripts/prepare_real_cmce.py --source flickr --out-dir data/real_matched \
  --n-total 100 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size tiny --epochs 8 --device cpu --seed 42 \
  --train-data data/real_matched/train.json --val-data data/real_matched/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v4
```

HF source: `jxie/flickr8k`. Images regenerated locally (not in git).

### 2.3 Reproduce COCO scaled tiny (coco_v2)

```bash
python scripts/prepare_real_cmce.py --source coco --out-dir data/real_coco_v2 \
  --n-total 400 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size tiny --epochs 20 --device cpu --seed 42 \
  --train-data data/real_coco_v2/train.json --val-data data/real_coco_v2/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_coco_v2
```

HF source: `phiyodr/coco2017`.

---

## 3. Takeaways (honest)

1. **Synthetic easy** is saturated on chunking; token-level InfoNCE fixed early alignment collapse (0.29→0.82).
2. **Hard synthetic** improves with curriculum (~0.14→~0.38) but tiny model stalls below 0.5.
3. **Picsum ≠ matched data**; CLIP on random images can hurt vs spatial priors.
4. **Matched Flickr/COCO** enable real pipeline checks; tiny+CLIP still ~0.25–0.32 align.
5. **Scaling width (small)** or **n/epochs alone** did not beat Flickr tiny v4 on alignment under this protocol.
6. Next research steps (not done): detector/phrase grounding with true boxes, larger compute, or official CMCE release format.

---

## 4. Results index

Committed metrics live under:

- `results/tiny_baseline/`
- `results/tiny_align_v2/`
- `results/tiny_hard_v{1,2,3}/`
- `results/tiny_real_v{1..6}/`
- `results/tiny_real_coco_v{1,2}/`

Each folder’s `metrics.json` is the source of truth for that run.

---

## 5. Pause note

Experiment iteration paused here by choice. Prefer extending **this document + metrics.json** over ad-hoc README sprawl when resuming.
