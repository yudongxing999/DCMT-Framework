#!/usr/bin/env python3
"""Tiny CPU-friendly DCMT training with supervised token-level alignment.

Loss = CE(labels) + w_contrast * contrastive + w_boundary * BoundaryLoss
     + w_token_align * token_InfoNCE(alignment, pairs)

Default out: results/tiny_align_v2, epochs=8.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmt.boundary_detector import BoundaryLoss
from dcmt.model import DCMTConfig, DCMTModel
from evaluate import CMCEEvaluator, load_cmce_json, simple_tokenize

# Hard-coded previous baseline for comparison in metrics.json
BASELINE_REF = {
    "chunk_f1": 0.9999999949999999,
    "alignment_acc": 0.28791666666666665,
    "cmce_score": 0.5727499979999999,
    "source": "results/tiny_baseline/metrics.json",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_target_matrix(
    pairs: List[Dict[str, int]],
    n_v: int,
    n_t: int,
) -> torch.Tensor:
    """Dense [N_v, N_t] float target with 1.0 at positive (v,t) pairs."""
    mat = torch.zeros(n_v, n_t, dtype=torch.float32)
    for p in pairs:
        v, t = int(p["v"]), int(p["t"])
        if 0 <= v < n_v and 0 <= t < n_t:
            mat[v, t] = 1.0
    return mat


class SyntheticCMCEDataset(Dataset):
    """Loads synthetic JSON with labels, boundaries, and alignment_token_pairs."""

    def __init__(
        self,
        path: str,
        img_size: int = 64,
        vocab_size: int = 1000,
        max_seq: int = 64,
        patch_size: int = 16,
    ):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        num_patches = (img_size // patch_size) ** 2
        self.n_v = 1 + num_patches  # CLS + patches
        self.n_t = max_seq

        self.samples: List[Dict[str, Any]] = []
        for i, item in enumerate(raw):
            text = item.get("text", "")
            input_ids, attention_mask = simple_tokenize(
                text, vocab_size=vocab_size, max_length=max_seq
            )
            g = torch.Generator().manual_seed(i + 42)
            image = torch.randn(3, img_size, img_size, generator=g)

            targets = item.get("text_boundary_targets")
            if targets is None:
                targets = [0] * max_seq
            if len(targets) < max_seq:
                targets = list(targets) + [0] * (max_seq - len(targets))
            targets = targets[:max_seq]

            label = int(item.get("label", 0))
            pairs = item.get("alignment_token_pairs", [])
            target_matrix = build_target_matrix(pairs, self.n_v, self.n_t)

            self.samples.append(
                {
                    "id": item.get("id", f"sample_{i}"),
                    "image": image,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "text_boundary_targets": torch.tensor(targets, dtype=torch.float32),
                    "label": torch.tensor(label, dtype=torch.long),
                    "alignment_token_pairs": pairs,
                    "align_target": target_matrix,
                    "annotations": {
                        "visual_chunks": item.get("visual_chunks", []),
                        "text_chunks": item.get("text_chunks", []),
                        "alignments": item.get("alignments", []),
                        "alignment_token_pairs": pairs,
                    },
                    "text": text,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_train(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "labels": torch.stack([b["label"] for b in batch], dim=0),
        "text_boundary_targets": torch.stack(
            [b["text_boundary_targets"] for b in batch], dim=0
        ),
        "align_target": torch.stack([b["align_target"] for b in batch], dim=0),
    }


def token_align_infonce(
    scores: torch.Tensor,
    target_matrix: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE over text dimension for each positive (v, t).

    scores: [B, N_v, N_t] cosine similarities in [-1, 1]
    target_matrix: [B, N_v, N_t] with 1 at positives
    text_mask: [B, N_t] validity (1=keep); pads masked out
    """
    B, Nv, Nt = scores.shape
    logits = scores / max(temperature, 1e-6)

    if text_mask is not None:
        pad = text_mask[:, None, :] == 0
        logits = logits.masked_fill(pad, float("-inf"))

    losses: List[torch.Tensor] = []
    pos_mask = target_matrix > 0.5
    for b in range(B):
        for v in range(Nv):
            pos_idx = pos_mask[b, v].nonzero(as_tuple=False).view(-1)
            if pos_idx.numel() == 0:
                continue
            row = logits[b, v]
            if not torch.isfinite(row).any():
                continue
            for t in pos_idx:
                t = t.long()
                if text_mask is not None and text_mask[b, t] == 0:
                    continue
                losses.append(F.cross_entropy(row.unsqueeze(0), t.unsqueeze(0)))

    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


def train_one_epoch(
    model: DCMTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    boundary_loss_fn: BoundaryLoss,
    device: str,
    boundary_weight: float = 0.5,
    token_align_weight: float = 1.0,
    contrast_weight: float = 0.1,
    token_align_temperature: float = 0.07,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_bd = 0.0
    total_ta = 0.0
    total_contrast = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        targets = batch["text_boundary_targets"].to(device)
        align_target = batch["align_target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            images,
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_chunks=True,
            return_alignment=True,
        )
        with torch.no_grad():
            ce = F.cross_entropy(out.logits, labels)
        if out.loss is not None:
            contrast_live = (out.loss - F.cross_entropy(out.logits, labels)) / 0.1
        else:
            contrast_live = out.logits.new_zeros(())

        text_boundaries = out.chunks["text"]
        L = min(text_boundaries.shape[1], targets.shape[1], attention_mask.shape[1])
        bd_loss = boundary_loss_fn(
            text_boundaries[:, :L],
            targets[:, :L],
            attention_mask[:, :L].float(),
        )

        scores = out.alignment
        Nv = scores.shape[1]
        Nt = scores.shape[2]
        tgt = align_target[:, :Nv, :Nt]
        if tgt.shape[1] < Nv or tgt.shape[2] < Nt:
            pad_v = Nv - tgt.shape[1]
            pad_t = Nt - tgt.shape[2]
            tgt = F.pad(tgt, (0, pad_t, 0, pad_v))

        ta_loss = token_align_infonce(
            scores[:, :, : attention_mask.shape[1]],
            tgt[:, :, : attention_mask.shape[1]],
            text_mask=attention_mask,
            temperature=token_align_temperature,
        )

        ce_live = F.cross_entropy(out.logits, labels)
        loss = (
            ce_live
            + contrast_weight * contrast_live
            + boundary_weight * bd_loss
            + token_align_weight * ta_loss
        )
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.detach()) * bs
        total_ce += float(ce_live.detach()) * bs
        total_bd += float(bd_loss.detach()) * bs
        total_ta += float(ta_loss.detach()) * bs
        total_contrast += float(contrast_live.detach()) * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "ce": total_ce / max(n, 1),
        "boundary": total_bd / max(n, 1),
        "token_align": total_ta / max(n, 1),
        "contrastive": total_contrast / max(n, 1),
    }


def build_tiny_config() -> DCMTConfig:
    return DCMTConfig(
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


def main():
    parser = argparse.ArgumentParser(description="Tiny DCMT CMCE trainer with token align")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="results/tiny_align_v2")
    parser.add_argument("--train-data", type=str, default="data/synthetic_train.json")
    parser.add_argument("--val-data", type=str, default="data/synthetic_val.json")
    parser.add_argument("--ce-weight", type=float, default=1.0, help="unused; CE always 1.0")
    parser.add_argument("--contrast-weight", type=float, default=0.1)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--token-align-weight", type=float, default=1.0)
    parser.add_argument("--token-align-temperature", type=float, default=0.07)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_tiny_config()
    model = DCMTModel(config).to(device)
    boundary_loss_fn = BoundaryLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_ds = SyntheticCMCEDataset(
        args.train_data,
        img_size=config.img_size,
        vocab_size=config.vocab_size,
        max_seq=config.max_seq_length,
        patch_size=config.patch_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_train,
        num_workers=0,
    )

    history: List[Dict[str, float]] = []
    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"Training tiny DCMT align_v2: params={nparams:,} epochs={args.epochs} "
        f"device={device} seed={args.seed} "
        f"w_contrast={args.contrast_weight} w_bd={args.boundary_weight} "
        f"w_token_align={args.token_align_weight}"
    )
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            boundary_loss_fn,
            device,
            boundary_weight=args.boundary_weight,
            token_align_weight=args.token_align_weight,
            contrast_weight=args.contrast_weight,
            token_align_temperature=args.token_align_temperature,
        )
        stats["epoch"] = float(epoch)
        history.append(stats)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"loss={stats['loss']:.4f}  ce={stats['ce']:.4f}  "
            f"bd={stats['boundary']:.4f}  ta={stats['token_align']:.4f}  "
            f"ctr={stats['contrastive']:.4f}"
        )

    ckpt_path = out_dir / "model.pt"
    model.save_pretrained(str(ckpt_path))
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Saved checkpoint -> {ckpt_path} ({ckpt_size_mb:.2f} MB)")

    val_dataset = load_cmce_json(
        args.val_data, img_size=config.img_size, vocab_size=config.vocab_size
    )
    evaluator = CMCEEvaluator(
        img_size=config.img_size, patch_size=config.patch_size
    )
    metrics = evaluator.evaluate(model, val_dataset, device=device)
    print(
        f"Val metrics: chunk_f1={metrics['chunk_f1']:.4f} "
        f"alignment_acc={metrics['alignment_acc']:.4f} "
        f"argmax_acc={metrics['argmax_acc']:.4f} "
        f"threshold_acc={metrics['threshold_acc']:.4f} "
        f"cmce_score={metrics['cmce_score']:.4f}"
    )

    delta_align = metrics["alignment_acc"] - BASELINE_REF["alignment_acc"]
    delta_cmce = metrics["cmce_score"] - BASELINE_REF["cmce_score"]

    payload = {
        "seed": args.seed,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss_weights": {
            "ce": 1.0,
            "contrastive": args.contrast_weight,
            "boundary": args.boundary_weight,
            "token_align": args.token_align_weight,
        },
        "token_align_temperature": args.token_align_temperature,
        "config": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "v_layers": config.v_layers,
            "patch_size": config.patch_size,
            "img_size": config.img_size,
            "vocab_size": config.vocab_size,
            "max_seq_length": config.max_seq_length,
            "num_labels": config.num_labels,
            "dropout": config.dropout,
        },
        "train_history": history,
        "train_loss_curve": [h["loss"] for h in history],
        "final_train_loss": history[-1]["loss"] if history else None,
        "metrics": {k: v for k, v in metrics.items()},
        "chunk_f1": metrics["chunk_f1"],
        "alignment_acc": metrics["alignment_acc"],
        "argmax_acc": metrics["argmax_acc"],
        "threshold_acc": metrics["threshold_acc"],
        "cmce_score": metrics["cmce_score"],
        "baseline_ref": BASELINE_REF,
        "delta_vs_baseline": {
            "alignment_acc": delta_align,
            "cmce_score": delta_cmce,
            "chunk_f1": metrics["chunk_f1"] - BASELINE_REF["chunk_f1"],
        },
        "checkpoint": str(ckpt_path),
        "checkpoint_size_mb": round(ckpt_size_mb, 3),
        "metric_notes": metrics.get("metric_notes", ""),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote metrics -> {metrics_path}")
    print(
        f"vs baseline alignment_acc: {BASELINE_REF['alignment_acc']:.4f} -> "
        f"{metrics['alignment_acc']:.4f} (delta={delta_align:+.4f})"
    )


if __name__ == "__main__":
    main()
