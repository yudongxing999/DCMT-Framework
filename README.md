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

Requirements: `torch`, `numpy`, `scipy`, `tqdm`.

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

## License

MIT — see `LICENSE`.

## Contact

Dongxing Yu — yudongxing@sandau.edu.cn
