#!/usr/bin/env python3
"""Tiny CPU-friendly DCMT training baseline on synthetic CMCE data.

Loss = CE(labels) + 0.1 * align_loss + 0.5 * BoundaryLoss(text_boundaries, targets, mask)

After training, evaluates on val with CMCEEvaluator and writes metrics.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticCMCEDataset(Dataset):
    """Loads synthetic JSON with labels + text_boundary_targets for training."""

    def __init__(
        self,
        path: str,
        img_size: int = 64,
        vocab_size: int = 1000,
        max_seq: int = 64,
    ):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

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

            self.samples.append(
                {
                    "id": item.get("id", f"sample_{i}"),
                    "image": image,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "text_boundary_targets": torch.tensor(targets, dtype=torch.float32),
                    "label": torch.tensor(label, dtype=torch.long),
                    "annotations": {
                        "visual_chunks": item.get("visual_chunks", []),
                        "text_chunks": item.get("text_chunks", []),
                        "alignments": item.get("alignments", []),
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
    }


def train_one_epoch(
    model: DCMTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    boundary_loss_fn: BoundaryLoss,
    device: str,
    boundary_weight: float = 0.5,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_bd = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        targets = batch["text_boundary_targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            images,
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_chunks=True,
            return_alignment=True,
        )
        base_loss = out.loss
        text_boundaries = out.chunks["text"]
        L = min(text_boundaries.shape[1], targets.shape[1], attention_mask.shape[1])
        bd_loss = boundary_loss_fn(
            text_boundaries[:, :L],
            targets[:, :L],
            attention_mask[:, :L].float(),
        )
        loss = base_loss + boundary_weight * bd_loss
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.detach()) * bs
        with torch.no_grad():
            ce = F.cross_entropy(out.logits, labels)
        total_ce += float(ce) * bs
        total_bd += float(bd_loss.detach()) * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "ce": total_ce / max(n, 1),
        "boundary": total_bd / max(n, 1),
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
    }


def main():
    parser = argparse.ArgumentParser(description="Tiny DCMT CMCE baseline trainer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="results/tiny_baseline")
    parser.add_argument("--train-data", type=str, default="data/synthetic_train.json")
    parser.add_argument("--val-data", type=str, default="data/synthetic_val.json")
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
    print(f"Training tiny DCMT: params={nparams:,} epochs={args.epochs} device={device} seed={args.seed}")
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model, train_loader, optimizer, boundary_loss_fn, device
        )
        stats["epoch"] = float(epoch)
        history.append(stats)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"loss={stats['loss']:.4f}  ce={stats['ce']:.4f}  bd={stats['boundary']:.4f}"
        )

    ckpt_path = out_dir / "model.pt"
    model.save_pretrained(str(ckpt_path))
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Saved checkpoint -> {ckpt_path} ({ckpt_size_mb:.2f} MB)")

    val_dataset = load_cmce_json(
        args.val_data, img_size=config.img_size, vocab_size=config.vocab_size
    )
    evaluator = CMCEEvaluator()
    metrics = evaluator.evaluate(model, val_dataset, device=device)
    print(
        f"Val metrics: chunk_f1={metrics['chunk_f1']:.4f} "
        f"alignment_acc={metrics['alignment_acc']:.4f} "
        f"cmce_score={metrics['cmce_score']:.4f}"
    )

    payload = {
        "seed": args.seed,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
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
        "metrics": metrics,
        "chunk_f1": metrics["chunk_f1"],
        "alignment_acc": metrics["alignment_acc"],
        "cmce_score": metrics["cmce_score"],
        "checkpoint": str(ckpt_path),
        "checkpoint_size_mb": round(ckpt_size_mb, 3),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
