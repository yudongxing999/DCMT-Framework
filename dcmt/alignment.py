"""
Cross-Modal Alignment Module

Implements contrastive learning for visual-linguistic correspondence
as described in Equation (3):
L_align = -log(exp(S(V_i, t_i) / τ) / Σ_j exp(S(V_i, t_j) / τ))

Author: Dongxing Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class CrossModalAlignment(nn.Module):
    """
    Cross-Modal Alignment Module

    Learns to align visual and textual representations using contrastive
    learning. Token-level scores are available via get_alignment_scores.
    """

    def __init__(
        self,
        d_model: int = 768,
        temperature: float = 0.07,
        projection_dim: int = 256
    ):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim

        self.visual_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

        # Ensure num_heads divides d_model
        n_heads = min(8, d_model)
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )

    def compute_similarity(
        self,
        visual: torch.Tensor,
        textual: torch.Tensor
    ) -> torch.Tensor:
        visual = F.normalize(visual, p=2, dim=-1)
        textual = F.normalize(textual, p=2, dim=-1)
        return torch.matmul(visual, textual.T)

    def contrastive_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        B = similarity.shape[0]
        temperature = self.log_temperature.exp()
        logits = similarity / temperature
        labels = torch.arange(B, device=similarity.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        return (loss_v2t + loss_t2v) / 2

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute cross-modal alignment with contrastive loss.

        Args:
            visual_features: [B, N_v, D]
            text_features: [B, N_t, D]
            visual_mask: Validity mask [B, N_v] (1=keep)
            text_mask: Validity mask [B, N_t] (1=keep)

        Returns:
            Dict with loss, batch similarity [B,B], attentions, and projected features
        """
        # Convert validity masks (1=keep) to key_padding_mask (True=ignore)
        text_kpm = (text_mask == 0) if text_mask is not None else None
        visual_kpm = (visual_mask == 0) if visual_mask is not None else None

        aligned_visual, v2t_attn = self.cross_attn_v2t(
            visual_features, text_features, text_features,
            key_padding_mask=text_kpm
        )
        aligned_text, t2v_attn = self.cross_attn_t2v(
            text_features, visual_features, visual_features,
            key_padding_mask=visual_kpm
        )

        if visual_mask is not None:
            visual_mask_expanded = visual_mask.unsqueeze(-1).float()
            visual_pooled = (aligned_visual * visual_mask_expanded).sum(1) / visual_mask_expanded.sum(1).clamp(min=1e-6)
        else:
            visual_pooled = aligned_visual.mean(dim=1)

        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(-1).float()
            text_pooled = (aligned_text * text_mask_expanded).sum(1) / text_mask_expanded.sum(1).clamp(min=1e-6)
        else:
            text_pooled = aligned_text.mean(dim=1)

        visual_proj = self.visual_proj(visual_pooled)
        text_proj = self.text_proj(text_pooled)

        similarity = self.compute_similarity(visual_proj, text_proj)
        loss = self.contrastive_loss(similarity)

        return {
            "loss": loss,
            "similarity": similarity,
            "v2t_attn": v2t_attn,
            "t2v_attn": t2v_attn,
            "aligned_visual": aligned_visual,
            "aligned_text": aligned_text,
            "visual_proj": visual_proj,
            "text_proj": text_proj,
        }

    def get_alignment_scores(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get token-level alignment scores between visual and text tokens.

        Args:
            visual_features: [B, N_v, D]
            text_features: [B, N_t, D]

        Returns:
            Alignment scores [B, N_v, N_t]
        """
        visual_norm = F.normalize(visual_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)
        return torch.bmm(visual_norm, text_norm.transpose(1, 2))


class MutualInformationEstimator(nn.Module):
    """
    Mutual Information Neural Estimation (MINE) for measuring
    cross-modal information sharing.
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        visual: torch.Tensor,
        textual: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate mutual information between visual and textual features.

        I(V; T) >= E[T(v, t)] - log(E[exp(T(v, t'))])
        """
        B = visual.shape[0]

        joint = torch.cat([visual, textual], dim=-1)
        joint_scores = self.network(joint)

        perm = torch.randperm(B, device=visual.device)
        textual_shuffled = textual[perm]
        marginal = torch.cat([visual, textual_shuffled], dim=-1)
        marginal_scores = self.network(marginal)

        # Keep log tensors on same device/dtype as visual
        log_B = torch.log(torch.tensor(B, dtype=visual.dtype, device=visual.device))
        log_2 = torch.log(torch.tensor(2.0, dtype=visual.dtype, device=visual.device))

        mi_estimate = joint_scores.mean() - torch.logsumexp(marginal_scores, dim=0) + log_B
        mi_bits = mi_estimate / log_2
        return mi_bits
