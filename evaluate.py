"""
CMCE Evaluation Script

Evaluates models on the Cross-Modal Chunking Evaluation benchmark.

Usage:
    python evaluate.py --data_path data/test_sample.json --device cpu

Author: Dongxing Yu
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from dcmt.model import DCMTModel, DCMTConfig


def simple_tokenize(text: str, vocab_size: int = 30522, max_length: int = 64):
    """Lightweight whitespace tokenizer for smoke / sample evaluation."""
    tokens = text.lower().split()
    ids = [((hash(tok) % (vocab_size - 1)) + 1) for tok in tokens][:max_length]
    if not ids:
        ids = [1]
    attention = [1] * len(ids)
    while len(ids) < max_length:
        ids.append(0)
        attention.append(0)
    return ids, attention


def visual_token_index(
    visual_chunk_id: int,
    n_visual: int,
    num_patches: int,
) -> int:
    """Map visual chunk id -> patch token index in [1, num_patches] (skip CLS=0)."""
    stride = max(1, num_patches // max(n_visual, 1))
    v = 1 + int(visual_chunk_id) * stride
    return int(min(max(v, 1), num_patches))


def text_chunk_to_token_index(
    text_chunk_id: int,
    text_chunks: List[Dict[str, Any]],
    n_t: int,
    approx_chars_per_tok: int = 5,
) -> int:
    """Map text_chunk id -> token index via token_start or char start//approx."""
    if not text_chunks:
        return min(int(text_chunk_id), max(n_t - 1, 0))
    chunk = None
    for c in text_chunks:
        if int(c.get("id", -1)) == int(text_chunk_id):
            chunk = c
            break
    if chunk is None and 0 <= int(text_chunk_id) < len(text_chunks):
        chunk = text_chunks[int(text_chunk_id)]
    if chunk is None:
        return min(int(text_chunk_id), max(n_t - 1, 0))
    if "token_start" in chunk:
        return int(min(max(int(chunk["token_start"]), 0), max(n_t - 1, 0)))
    start = int(chunk.get("start", 0))
    tok_idx = start // max(approx_chars_per_tok, 1)
    return int(min(max(tok_idx, 0), max(n_t - 1, 0)))


def resolve_token_pairs(
    annotations: Dict[str, Any],
    n_v: int,
    n_t: int,
    img_size: int = 64,
    patch_size: int = 16,
) -> List[Tuple[int, int]]:
    """
    Resolve GT (v, t) token pairs for alignment scoring.

    Prefer explicit alignment_token_pairs; else map chunk ids using
    text_chunks starts and visual stride over patches.
    """
    pairs_raw = annotations.get("alignment_token_pairs")
    if pairs_raw:
        out = []
        for p in pairs_raw:
            v = int(p["v"])
            t = int(p["t"])
            if 0 <= v < n_v and 0 <= t < n_t:
                out.append((v, t))
        if out:
            return out

    alignments = annotations.get("alignments", [])
    text_chunks = annotations.get("text_chunks", [])
    visual_chunks = annotations.get("visual_chunks", [])
    n_visual = max(len(visual_chunks), 1)
    primary = [a for a in alignments if a.get("primary", True)]
    if not primary:
        primary = alignments

    num_patches = (img_size // patch_size) ** 2
    effective_patches = num_patches
    if n_v == num_patches + 1:
        effective_patches = num_patches
    elif n_v > 1:
        effective_patches = n_v - 1

    out = []
    for align in primary:
        vid = int(align["visual_chunk"])
        tid = int(align["text_chunk"])
        v = visual_token_index(vid, n_visual, effective_patches)
        t = text_chunk_to_token_index(tid, text_chunks, n_t)
        v = min(max(v, 0), n_v - 1)
        t = min(max(t, 0), n_t - 1)
        out.append((v, t))
    return out


def load_cmce_json(path: str, img_size: int = 224, vocab_size: int = 30522) -> List[Dict[str, Any]]:
    """
    Inline JSON loader matching data/test_sample.json schema.

    Each sample is expected to have:
      - text: str
      - visual_chunks, text_chunks, alignments (annotations)
      - optional alignment_token_pairs (token-level GT)
      - optional image_path (synthetic image used if file missing)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for i, item in enumerate(raw):
        text = item.get("text", "")
        input_ids, attention_mask = simple_tokenize(text, vocab_size=vocab_size)

        image_path = item.get("image_path")
        image = None
        if image_path:
            full = Path(path).parent / image_path
            if full.exists():
                try:
                    from PIL import Image
                    import torchvision.transforms as T
                    img = Image.open(full).convert("RGB")
                    image = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                    ])(img)
                except Exception:
                    image = None
        if image is None:
            g = torch.Generator().manual_seed(i + 42)
            image = torch.randn(3, img_size, img_size, generator=g)

        token_pairs = item.get("alignment_token_pairs", [])
        samples.append({
            "id": item.get("id", f"sample_{i}"),
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "alignment_token_pairs": token_pairs,
            "annotations": {
                "visual_chunks": item.get("visual_chunks", []),
                "text_chunks": item.get("text_chunks", []),
                "alignments": item.get("alignments", []),
                "alignment_token_pairs": token_pairs,
            },
            "text": text,
            "metadata": item.get("metadata", {}),
        })
    return samples


class CMCEEvaluator:
    """Evaluator for the Cross-Modal Chunking Evaluation benchmark."""

    def __init__(
        self,
        chunk_threshold: float = 0.5,
        alignment_threshold: float = 0.5,
        img_size: int = 64,
        patch_size: int = 16,
    ):
        self.chunk_threshold = chunk_threshold
        self.alignment_threshold = alignment_threshold
        self.img_size = img_size
        self.patch_size = patch_size

    def compute_chunk_f1(
        self,
        pred_boundaries: np.ndarray,
        gt_boundaries: np.ndarray,
        tolerance: int = 2
    ) -> Dict[str, float]:
        pred_pos = np.where(pred_boundaries > self.chunk_threshold)[0]
        gt_pos = np.where(gt_boundaries > 0.5)[0]

        if len(pred_pos) == 0 and len(gt_pos) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if len(pred_pos) == 0 or len(gt_pos) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        matched_pred = set()
        matched_gt = set()
        for p in pred_pos:
            for g in gt_pos:
                if abs(int(p) - int(g)) <= tolerance and g not in matched_gt:
                    matched_pred.add(p)
                    matched_gt.add(g)
                    break

        precision = len(matched_pred) / len(pred_pos)
        recall = len(matched_gt) / len(gt_pos)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"precision": precision, "recall": recall, "f1": f1}

    def compute_alignment_accuracy(
        self,
        pred_alignment: np.ndarray,
        gt_alignment: list,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Token-level alignment accuracy.

        Args:
            pred_alignment: Token-level alignment matrix [N_v, N_t] (cosine in [-1,1])
            gt_alignment: List of {visual_chunk, text_chunk, ...} (fallback)
            annotations: Full annotation dict; prefer alignment_token_pairs / text_chunks

        Returns:
            Dict with threshold_acc, argmax_acc, alignment_acc (primary = argmax_acc).
        """
        if pred_alignment.ndim != 2:
            raise ValueError(f"Expected [N_v, N_t] alignment, got shape {pred_alignment.shape}")

        n_v, n_t = pred_alignment.shape
        ann = dict(annotations or {})
        if "alignments" not in ann and gt_alignment is not None:
            ann["alignments"] = gt_alignment

        pairs = resolve_token_pairs(
            ann, n_v, n_t, img_size=self.img_size, patch_size=self.patch_size
        )
        if not pairs:
            return {
                "threshold_acc": 0.0,
                "argmax_acc": 0.0,
                "alignment_acc": 0.0,
            }

        thr_correct = 0
        arg_correct = 0
        for v, t in pairs:
            score = float(pred_alignment[v, t])
            if score > self.alignment_threshold:
                thr_correct += 1
            pred_t = int(np.argmax(pred_alignment[v, :]))
            if pred_t == int(t):
                arg_correct += 1

        total = len(pairs)
        threshold_acc = thr_correct / total
        argmax_acc = arg_correct / total
        alignment_acc = argmax_acc
        return {
            "threshold_acc": threshold_acc,
            "argmax_acc": argmax_acc,
            "alignment_acc": alignment_acc,
            "alignment_acc_max": max(threshold_acc, argmax_acc),
        }

    def evaluate(
        self,
        model,
        dataset,
        device: str = "cpu"
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        all_chunk_f1 = []
        all_threshold_acc = []
        all_argmax_acc = []
        all_alignment_acc = []

        cfg = getattr(model, "config", None)
        if cfg is not None:
            self.img_size = getattr(cfg, "img_size", self.img_size)
            self.patch_size = getattr(cfg, "patch_size", self.patch_size)

        with torch.no_grad():
            for sample in tqdm(dataset, desc="Evaluating"):
                image = sample["image"].unsqueeze(0).to(device)
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = sample.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(0).to(device)

                output = model(
                    image,
                    input_ids,
                    attention_mask=attention_mask,
                    return_chunks=True,
                    return_alignment=True,
                )

                visual_boundaries = output.chunks["visual"][0].cpu().numpy()
                text_boundaries = output.chunks["text"][0].cpu().numpy()
                alignment = output.alignment[0].cpu().numpy()

                gt = sample["annotations"]
                if sample.get("alignment_token_pairs") and not gt.get("alignment_token_pairs"):
                    gt = dict(gt)
                    gt["alignment_token_pairs"] = sample["alignment_token_pairs"]

                text_len = len(text_boundaries)
                gt_boundary = np.zeros(text_len, dtype=np.float32)
                for c in gt.get("text_chunks", []):
                    if "token_start" in c:
                        tok_idx = int(c["token_start"])
                    else:
                        start = int(c.get("start", 0))
                        tok_idx = start // 5
                    tok_idx = min(max(tok_idx, 0), text_len - 1)
                    gt_boundary[tok_idx] = 1.0

                attn = sample.get("attention_mask")
                if attn is not None:
                    valid = int(attn.sum().item()) if hasattr(attn, "sum") else int(sum(attn))
                    valid = max(valid, 1)
                    chunk_metrics = self.compute_chunk_f1(
                        text_boundaries[:valid], gt_boundary[:valid]
                    )
                else:
                    chunk_metrics = self.compute_chunk_f1(text_boundaries, gt_boundary)
                all_chunk_f1.append(chunk_metrics["f1"])

                align_metrics = self.compute_alignment_accuracy(
                    alignment, gt.get("alignments", []), annotations=gt
                )
                all_threshold_acc.append(align_metrics["threshold_acc"])
                all_argmax_acc.append(align_metrics["argmax_acc"])
                all_alignment_acc.append(align_metrics["alignment_acc"])

        results = {
            "chunk_f1": float(np.mean(all_chunk_f1)) if all_chunk_f1 else 0.0,
            "threshold_acc": float(np.mean(all_threshold_acc)) if all_threshold_acc else 0.0,
            "argmax_acc": float(np.mean(all_argmax_acc)) if all_argmax_acc else 0.0,
            "alignment_acc": float(np.mean(all_alignment_acc)) if all_alignment_acc else 0.0,
        }
        results["alignment_acc_max"] = max(results["threshold_acc"], results["argmax_acc"])
        results["cmce_score"] = 0.4 * results["chunk_f1"] + 0.6 * results["alignment_acc"]
        results["metric_notes"] = (
            "alignment_acc is argmax_acc (pred_t = argmax_t S[v,t] == gt t); "
            "threshold_acc uses score > alignment_threshold on mapped (v,t); "
            "pairs from alignment_token_pairs when present, else chunk→token mapping"
        )
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on CMCE benchmark")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--data_path", type=str, default="data/test_sample.json", help="Path to test JSON")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--tiny", action="store_true", help="Use tiny randomly-initialized model (CPU smoke)")
    args = parser.parse_args()

    if args.model_path:
        model = DCMTModel.from_pretrained(args.model_path)
        img_size = model.config.img_size
        vocab_size = model.config.vocab_size
    elif args.tiny:
        config = DCMTConfig(
            d_model=64,
            n_heads=4,
            n_layers=1,
            d_ff=128,
            v_layers=1,
            patch_size=16,
            img_size=64,
            vocab_size=1000,
            max_seq_length=64,
            num_labels=10,
            dropout=0.0,
        )
        model = DCMTModel(config)
        img_size = 64
        vocab_size = 1000
    else:
        config = DCMTConfig()
        model = DCMTModel(config)
        img_size = config.img_size
        vocab_size = config.vocab_size

    dataset = load_cmce_json(args.data_path, img_size=img_size, vocab_size=vocab_size)

    evaluator = CMCEEvaluator(img_size=img_size, patch_size=getattr(model.config, "patch_size", 16))
    results = evaluator.evaluate(model, dataset, args.device)

    print("\n" + "=" * 50)
    print("CMCE Evaluation Results")
    print("=" * 50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "cmce_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path / 'cmce_results.json'}")


if __name__ == "__main__":
    main()
