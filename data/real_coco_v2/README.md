# real_coco_v2

Matched `coco` CMCE split for scale experiments (CLIP grounding).

- n_train=320, n_val=80, n_total=400
- seed=42, max_caption_tokens=12
- grounding=clip, clip=transformers:openai/clip-vit-base-patch32

Regenerate images with `scripts/prepare_real_cmce.py` (images/ not committed).
See `meta.json` for spotcheck captions.
