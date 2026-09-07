"""
Adaptive Boundary Detection Module

Implements context-sensitive token boundary detection as described in Equation (1):
B(x; θ) = σ(f_θ(x) - T)

Author: Dongxing Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdaptiveBoundaryDetector(nn.Module):
    """
    Adaptive Boundary Detection for Dynamic Tokenization

    This module learns to predict token boundaries based on semantic coherence,
    mimicking human cognitive chunking mechanisms.

    The boundary detection function:
        B(x; θ) = σ(f_θ(x) - T)

    Args:
        d_model: Model dimensionality (default: 768)
        threshold: Boundary detection threshold T (default: 0.5)
        hidden_dim: Hidden layer dimension (default: 256)
    """

    def __init__(
        self,
        d_model: int = 768,
        threshold: float = 0.5,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold

        self.boundary_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        n_heads = min(8, d_model)
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.context_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )

        self.threshold_adjust = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Detect token boundaries in the input sequence.

        Args:
            x: Input features [batch_size, seq_length, d_model]
            mask: Optional validity mask [batch_size, seq_length] (1=keep, 0=pad)

        Returns:
            boundary_probs: Boundary probabilities [batch_size, seq_length]
        """
        B, L, D = x.shape

        # PyTorch key_padding_mask: True = ignore; convert from validity mask
        key_padding_mask = (mask == 0) if mask is not None else None
        context_x, _ = self.context_attn(x, x, x, key_padding_mask=key_padding_mask)

        combined = x + context_x

        boundary_scores = self.boundary_net(combined).squeeze(-1)  # [B, L]

        effective_threshold = self.threshold + self.threshold_adjust
        boundary_probs = torch.sigmoid(boundary_scores - effective_threshold)

        if mask is not None:
            boundary_probs = boundary_probs * mask.float()

        return boundary_probs

    def get_hard_boundaries(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get discrete boundary predictions (0 or 1)."""
        probs = self.forward(x, mask)
        return (probs > 0.5).float()

    def get_chunks(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segment input into chunks based on detected boundaries.

        Returns:
            chunk_ids: Chunk assignment for each position [batch_size, seq_length]
            num_chunks: Number of chunks per batch item [batch_size]
        """
        boundaries = self.get_hard_boundaries(x, mask)
        chunk_ids = boundaries.cumsum(dim=-1).long()
        num_chunks = chunk_ids.max(dim=-1).values + 1
        return chunk_ids, num_chunks


class BoundaryLoss(nn.Module):
    """
    Loss function for boundary detection training.

    Combines BCE with ground truth, smoothness regularization, and
    minimum chunk size constraint.
    """

    def __init__(
        self,
        smoothness_weight: float = 0.1,
        min_chunk_size: int = 2
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.min_chunk_size = min_chunk_size

    def forward(
        self,
        pred_boundaries: torch.Tensor,
        target_boundaries: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(pred_boundaries)

        bce_loss = F.binary_cross_entropy(
            pred_boundaries * mask,
            target_boundaries * mask,
            reduction="sum"
        ) / mask.sum()

        diff = torch.abs(pred_boundaries[:, 1:] - pred_boundaries[:, :-1])
        smoothness_loss = (diff * mask[:, 1:]).sum() / mask[:, 1:].sum()

        return bce_loss + self.smoothness_weight * smoothness_loss
