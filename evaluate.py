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
from typing import Dict, Any, List, Optional
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


def load_cmce_json(path: str, img_size: int = 224, vocab_size: int = 30522) -> List[Dict[str, Any]]:
    """
    Inline JSON loader matching data/test_sample.json schema.

    Each sample is expected to have:
      - text: str
      - visual_chunks, text_chunks, alignments (annotations)
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
            # Deterministic synthetic image for reviewer smoke runs
            g = torch.Generator().manual_seed(i + 42)
            image = torch.randn(3, img_size, img_size, generator=g)

        samples.append({
            "id": item.get("id", f"sample_{i}"),
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "annotations": {
                "visual_chunks": item.get("visual_chunks", []),
                "text_chunks": item.get("text_chunks", []),
                "alignments": item.get("alignments", []),
            },
            "text": text,
        })
    return samples


class CMCEEvaluator:
    """Evaluator for the Cross-Modal Chunking Evaluation benchmark."""

    def __init__(
        self,
        chunk_threshold: float = 0.5,
        alignment_threshold: float = 0.5
    ):
        self.chunk_threshold = chunk_threshold
        self.alignment_threshold = alignment_threshold

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
        gt_alignment: list
    ) -> float:
        """
        Args:
            pred_alignment: Token-level alignment matrix [N_v, N_t]
            gt_alignment: List of {visual_chunk, text_chunk, ...}
        """
        if pred_alignment.ndim != 2:
            raise ValueError(f"Expected [N_v, N_t] alignment, got shape {pred_alignment.shape}")

        n_v, n_t = pred_alignment.shape
        correct = 0
        total = len(gt_alignment)

        for align in gt_alignment:
            v_idx = int(align["visual_chunk"])
            t_idx = int(align["text_chunk"])
            # Map chunk indices into token matrix bounds when chunk ids are small
            v = min(v_idx, n_v - 1)
            t = min(t_idx, n_t - 1)
            if pred_alignment[v, t] > self.alignment_threshold:
                correct += 1

        return correct / total if total > 0 else 0.0

    def evaluate(
        self,
        model,
        dataset,
        device: str = "cpu"
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        all_chunk_f1 = []
        all_alignment_acc = []

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

                # chunks is a dict with unequal-length visual/text boundaries
                visual_boundaries = output.chunks["visual"][0].cpu().numpy()
                text_boundaries = output.chunks["text"][0].cpu().numpy()
                # alignment is token-level [B, N_v, N_t]
                alignment = output.alignment[0].cpu().numpy()

                gt = sample["annotations"]
                text_len = len(text_boundaries)
                gt_boundary = np.zeros(text_len, dtype=np.float32)
                for c in gt.get("text_chunks", []):
                    start = int(c.get("start", 0))
                    # Map character start loosely into token index space
                    tok_idx = min(start // 5, text_len - 1)
                    gt_boundary[tok_idx] = 1.0

                chunk_metrics = self.compute_chunk_f1(text_boundaries, gt_boundary)
                all_chunk_f1.append(chunk_metrics["f1"])

                align_acc = self.compute_alignment_accuracy(alignment, gt.get("alignments", []))
                all_alignment_acc.append(align_acc)

        results = {
            "chunk_f1": float(np.mean(all_chunk_f1)) if all_chunk_f1 else 0.0,
            "alignment_acc": float(np.mean(all_alignment_acc)) if all_alignment_acc else 0.0,
        }
        results["cmce_score"] = 0.4 * results["chunk_f1"] + 0.6 * results["alignment_acc"]
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

    evaluator = CMCEEvaluator()
    results = evaluator.evaluate(model, dataset, args.device)

    print("\n" + "=" * 50)
    print("CMCE Evaluation Results")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "cmce_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path / 'cmce_results.json'}")


if __name__ == "__main__":
    main()
