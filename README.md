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

# either
pip install -r requirements.txt
pip install -e .

# or editable install alone (pulls deps from pyproject.toml)
pip install -e .
```

Requirements: `torch`, `numpy`, `scipy`, `tqdm`, `Pillow`, `torchvision` (image load).

## Package layout

```
dcmt/
  __init__.py
  model.py              # DCMTModel, DCMTConfig, DCMTOutput
  boundary_detector.py
  hierarchical.py
  alignment.py
evaluate.py
statistical_tests.py
statistical_details.csv
requirements.txt
pyproject.toml
README.md
tests/test_forward_smoke.py
data/test_sample.json
data/synthetic_train.json
data/synthetic_val.json
scripts/make_synthetic_cmce.py
scripts/prepare_real_cmce.py
scripts/train_tiny.py
data/README_synthetic.md
results/tiny_baseline/metrics.json
results/tiny_align_v2/metrics.json
```

## Usage

```python
from dcmt import DCMTModel, DCMTConfig
import torch

config = DCMTConfig()          # default ViT/BERT-scale dims
model = DCMTModel(config)

images = torch.randn(1, 3, 224, 224)
input_ids = torch.randint(0, 30522, (1, 32))
out = model(images, input_ids)

print(out.logits.shape)                 # [B, num_labels]
print(out.chunks["visual"].shape)       # [B, N_v] boundary probs
print(out.chunks["text"].shape)         # [B, N_t] (not stacked; lengths may differ)
print(out.alignment.shape)              # [B, N_v, N_t] token-level scores
```

## Reviewer quickstart (CPU smoke)

No pretrained weights or downloads required — modules are randomly initialized.

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests/test_forward_smoke.py -q
# or
python tests/test_forward_smoke.py
```

Optional tiny evaluation over the sample JSON (synthetic images if files missing):

```bash
python evaluate.py --tiny --device cpu --data_path data/test_sample.json
```

The default full-size config (`d_model=768`, 12 layers) also builds without
pretrained weights and `forward()` runs on random tensors, but is heavy on CPU.
Prefer the tiny config in the smoke test / `--tiny` flag for reviewers.

## Bug fixes in this migration

1. Proper installable `dcmt/` package with absolute imports (`from dcmt...`).
2. Chunk boundaries returned as `{"visual": ..., "text": ...}` (no `torch.stack` of unequal lengths).
3. Evaluation gets token-level alignment `[B, N_v, N_t]` via `get_alignment_scores`; contrastive loss still computed in alignment forward.
4. `MutualInformationEstimator` log tensors use the same device/dtype as inputs.
5. `key_padding_mask = (mask == 0)` when `mask` is a validity mask (`1=keep`).
6. `evaluate.py` imports `dcmt.model`, loads `data/test_sample.json` inline, and indexes alignment as `[N_v, N_t]`.

## Tiny training baseline

CPU-friendly synthetic CMCE training to verify the full train→eval loop.

```bash
# from repo root (after pip install -e . or with PYTHONPATH=.)
python scripts/make_synthetic_cmce.py --seed 42
python scripts/train_tiny.py --epochs 5 --device cpu --seed 42
```

This writes:
- `data/synthetic_train.json` / `data/synthetic_val.json` (200 / 40 samples)
- `results/tiny_baseline/model.pt` (tiny checkpoint via `save_pretrained`)
- `results/tiny_baseline/metrics.json` (loss curve + CMCE metrics)

**Metrics** (from `CMCEEvaluator` in `evaluate.py`):
- `chunk_f1` — F1 of predicted text token boundaries vs GT chunk starts (tolerance ±2 tokens)
- `threshold_acc` — fraction of GT token pairs with score > 0.5
- `argmax_acc` — fraction where `argmax` over text dim for visual token `v` equals GT `t`
- `alignment_acc` — **primary** = `argmax_acc` (prefer over threshold; cosine scale is poorly calibrated)
- `cmce_score` — `0.4 * chunk_f1 + 0.6 * alignment_acc`

Tiny config: `d_model=64`, 1 layer, `img_size=64`, `vocab_size=1000`, `max_seq_length=64`, `num_labels=10`.
Baseline loss: `CE(labels) + 0.1 * contrastive + 0.5 * BoundaryLoss`.

### align_v2 (token-level supervised alignment)

Improves token-matrix alignment by supervising `output.alignment` with primary
`alignment_token_pairs` from the synthetic generator (InfoNCE over text dim per
positive visual token). Secondary noisy chunk pairs are excluded from supervision.

```bash
python scripts/make_synthetic_cmce.py --seed 42
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 --out results/tiny_align_v2
```

Loss weights (CLI): `CE + --contrast-weight 0.1 + --boundary-weight 0.5 + --token-align-weight 1.0`.

Expected keys in `results/tiny_align_v2/metrics.json`:
`chunk_f1`, `alignment_acc`, `argmax_acc`, `threshold_acc`, `cmce_score`,
`baseline_ref`, `delta_vs_baseline`.

Synthetic JSON includes `alignment_token_pairs: [{"v": int, "t": int}, ...]`
(`v` skips CLS=0; `t` is text chunk start token). Regenerate with
`scripts/make_synthetic_cmce.py` if files are missing.


### Harder synthetic (`tiny_hard_v1`)

```bash
python scripts/make_synthetic_cmce.py --seed 42 --difficulty hard --n-train 300 --n-val 60
python scripts/train_tiny.py --epochs 8 --device cpu --seed 42 \
  --train-data data/synthetic_hard_train.json \
  --val-data data/synthetic_hard_val.json \
  --out results/tiny_hard_v1
```

Hard mode adds longer multi-chunk texts, confusable object names, irregular
visual strides, boundary noise, and distractor alignments (annotation-only).
Expect **lower** `alignment_acc` / `cmce_score` than easy `tiny_align_v2`
(alignment_acc≈0.817, cmce≈0.890) — that is intentional.

See `data/README_synthetic.md`.

## Real image-text pipeline (tiny / smoke)

```bash
pip install -r requirements.txt   # includes Pillow, torchvision
python scripts/prepare_real_cmce.py --n-total 80 --seed 42
python scripts/train_tiny.py --epochs 5 --device cpu --seed 42 \
  --train-data data/real/train.json --val-data data/real/val.json \
  --out results/tiny_real_v1
```

Offline fallback if downloads fail:

```bash
python scripts/prepare_real_cmce.py --from-dir /path/to/images --n-total 60
```

**Caveats:** visual alignments are **pseudo** (patch indices spaced across the
image; no object detectors). Image binaries under `data/real/images/` are
re-downloaded by the script and generally not committed. See `data/real/README.md`.

## License


MIT — see `LICENSE`.

## Contact

Dongxing Yu — yudongxing@sandau.edu.cn
