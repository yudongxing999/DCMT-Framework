"""
Hierarchical Representation Module

Implements multi-level transformer encoders with bidirectional connections
as described in Equation (2):
h^l = TransformerBlock(h^{l-1} + TopDown(h^{l+1}))

Author: Dongxing Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class TopDownProjection(nn.Module):
    """Top-down projection for feedback connections from higher levels."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(
        self,
        lower: torch.Tensor,
        higher: torch.Tensor
    ) -> torch.Tensor:
        if higher.shape[1] != lower.shape[1]:
            higher = F.interpolate(
                higher.transpose(1, 2),
                size=lower.shape[1],
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

        projected = self.projection(higher)
        combined = torch.cat([lower, projected], dim=-1)
        gate = self.gate(combined)
        return gate * projected


class HierarchicalLevel(nn.Module):
    """Single level in the hierarchical representation."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.top_down = TopDownProjection(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        top_down_signal: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input representation h^{l-1} [B, L, D]
            top_down_signal: Higher-level representation h^{l+1}
            mask: Validity mask [B, L] (1=keep); converted to key_padding_mask
        """
        if top_down_signal is not None:
            td = self.top_down(x, top_down_signal)
            x = x + td

        # PyTorch key_padding_mask: True = ignore
        key_padding_mask = (mask == 0) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class HierarchicalRepresentation(nn.Module):
    """
    Hierarchical Representation Network with Bidirectional Connections

    Implements multi-level processing with bottom-up and top-down flow:
        h^l = TransformerBlock(h^{l-1} + TopDown(h^{l+1}))
    """

    def __init__(
        self,
        d_model: int = 768,
        n_levels: int = 3,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_levels = n_levels
        self.d_model = d_model

        # Ensure num_heads divides d_model
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.levels = nn.ModuleList([
            HierarchicalLevel(d_model, n_heads, d_ff, dropout)
            for _ in range(n_levels)
        ])

        self.chunk_pool = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh()
            )
            for _ in range(n_levels)
        ])

        self.output_proj = nn.Linear(d_model * n_levels, d_model)

    def forward(
        self,
        x: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_all_levels: bool = False
    ) -> torch.Tensor:
        B, L, D = x.shape

        level_outputs = []
        current = x
        for level in self.levels:
            current = level(current, top_down_signal=None, mask=mask)
            level_outputs.append(current)

        refined_outputs = [level_outputs[-1]]

        for i in range(self.n_levels - 2, -1, -1):
            top_down_signal = refined_outputs[-1]
            refined = self.levels[i](
                level_outputs[i],
                top_down_signal=top_down_signal,
                mask=mask
            )
            refined_outputs.append(refined)

        refined_outputs = refined_outputs[::-1]

        if return_all_levels:
            return refined_outputs

        pooled_outputs = []
        for i, (output, pool) in enumerate(zip(refined_outputs, self.chunk_pool)):
            if boundaries is not None:
                weights = 1.0 + 0.5 * boundaries.unsqueeze(-1)
                weighted = output * weights
                pooled = pool(weighted)
            else:
                pooled = pool(output)
            pooled_outputs.append(pooled)

        combined = torch.cat(pooled_outputs, dim=-1)
        output = self.output_proj(combined)
        return output

    def get_level_representations(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        return self.forward(x, boundaries=None, mask=mask, return_all_levels=True)
