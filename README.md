# DCMT Framework

**Dynamic Cross-Modal Tokenization** — adaptive token boundaries that integrate human chunking mechanisms into multimodal models.

Paper: *Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs*

**Experiment status:** paused. Full protocol, tables, takeaways, and reproduce commands → **[EXPERIMENTS.md](./EXPERIMENTS.md)**.

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
Optional CLIP grounding: `transformers>=4.36`.
Optional matched Flickr/COCO: `datasets` (`jxie/flickr8k`, `phiyodr/coco2017`).

## Quick smoke

```bash
python -m pytest tests/test_forward_smoke.py -q
```

Recommended real matched tiny run (best matched align so far ≈0.320):

```bash
pip install datasets transformers
python scripts/prepare_real_cmce.py --source flickr --out-dir data/real_matched \
  --n-total 100 --seed 42 --grounding clip --max-caption-tokens 12
python scripts/train_tiny.py --model-size tiny --epochs 8 --device cpu --seed 42 \
  --train-data data/real_matched/train.json --val-data data/real_matched/val.json \
  --token-align-weight 1.5 --visual-boundary-weight 0.25 \
  --out results/tiny_real_v4
```

See [EXPERIMENTS.md](./EXPERIMENTS.md) for synthetic easy/hard, COCO, scaling ablations, and caveats.

## License

MIT — see `LICENSE`.

## Contact

Dongxing Yu — yudongxing@sandau.edu.cn
