"""
Dynamic Cross-Modal Tokenization (DCMT) Model

This module implements the main DCMT architecture as described in:
"Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs"

Author: Dongxing Yu
Email: yudongxing@sandau.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from dcmt.boundary_detector import AdaptiveBoundaryDetector
from dcmt.hierarchical import HierarchicalRepresentation
from dcmt.alignment import CrossModalAlignment


@dataclass
class DCMTOutput:
    """Output container for DCMT model predictions."""
    logits: torch.Tensor
    chunks: Optional[Dict[str, torch.Tensor]] = None
    alignment: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    loss: Optional[torch.Tensor] = None


class DCMTConfig:
    """Configuration class for DCMT model."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        patch_size: int = 16,
        img_size: int = 224,
        v_layers: int = 12,
        boundary_threshold: float = 0.5,
        vocab_size: int = 30522,
        max_seq_length: int = 512,
        num_labels: int = 1000,
        **kwargs
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.patch_size = patch_size
        self.img_size = img_size
        self.v_layers = v_layers
        self.boundary_threshold = boundary_threshold
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels

        for key, value in kwargs.items():
            setattr(self, key, value)


class VisionEncoder(nn.Module):
    """Vision Transformer encoder for image processing."""

    def __init__(self, config: DCMTConfig):
        super().__init__()
        self.config = config

        self.patch_embed = nn.Conv2d(
            3, config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        num_patches = (config.img_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.v_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [batch_size, 3, img_size, img_size]
        Returns:
            Visual features [batch_size, num_patches+1, d_model]
        """
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)

        return x


class TextEncoder(nn.Module):
    """BERT-style text encoder."""

    def __init__(self, config: DCMTConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length] (1=keep)
        Returns:
            Text features [batch_size, seq_length, d_model]
        """
        B, L = input_ids.shape

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)

        # PyTorch key_padding_mask: True = ignore (pad)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        return x


class DCMTModel(nn.Module):
    """
    Dynamic Cross-Modal Tokenization Model

    Implements adaptive boundary detection, hierarchical representation,
    and cross-modal alignment for multimodal understanding.

    Example:
        >>> config = DCMTConfig()
        >>> model = DCMTModel(config)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> input_ids = torch.randint(0, 30522, (1, 32))
        >>> output = model(image, input_ids)
        >>> print(output.logits.shape)
    """

    def __init__(self, config: DCMTConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = VisionEncoder(config)
        self.text_encoder = TextEncoder(config)

        self.boundary_detector = AdaptiveBoundaryDetector(
            d_model=config.d_model,
            threshold=config.boundary_threshold
        )
        self.hierarchical = HierarchicalRepresentation(
            d_model=config.d_model,
            n_levels=3
        )
        self.cross_modal_alignment = CrossModalAlignment(
            d_model=config.d_model,
            temperature=0.07
        )

        self.classifier = nn.Linear(config.d_model * 2, config.num_labels)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_chunks: bool = True,
        return_alignment: bool = True,
    ) -> DCMTOutput:
        """
        Forward pass of DCMT model.

        Args:
            images: Input images [batch_size, 3, H, W]
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length] (1=keep)
            labels: Target labels for supervised training
            return_chunks: Whether to return chunk boundaries
            return_alignment: Whether to return token-level alignment scores

        Returns:
            DCMTOutput containing logits, chunks dict, token-level alignment, and optional loss
        """
        visual_features = self.vision_encoder(images)  # [B, N_v, D]
        text_features = self.text_encoder(input_ids, attention_mask)  # [B, N_t, D]

        # Adaptive boundary detection: B(x; θ) = σ(f_θ(x) - T)
        visual_boundaries = self.boundary_detector(visual_features)
        text_boundaries = self.boundary_detector(text_features)

        # Hierarchical representation with bidirectional connections
        visual_hierarchical = self.hierarchical(visual_features, visual_boundaries)
        text_hierarchical = self.hierarchical(text_features, text_boundaries)

        # Contrastive alignment loss (batch-level similarity)
        alignment_output = self.cross_modal_alignment(
            visual_hierarchical,
            text_hierarchical,
            text_mask=attention_mask,
        )

        # Token-level alignment [B, N_v, N_t] for evaluation
        token_alignment = None
        if return_alignment:
            token_alignment = self.cross_modal_alignment.get_alignment_scores(
                visual_hierarchical,
                text_hierarchical,
            )

        visual_pooled = visual_hierarchical.mean(dim=1)
        text_pooled = text_hierarchical.mean(dim=1)
        fused = torch.cat([visual_pooled, text_pooled], dim=-1)

        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            align_loss = alignment_output["loss"]
            loss = ce_loss + 0.1 * align_loss

        # Do NOT stack unequal-length visual/text boundaries
        chunks = None
        if return_chunks:
            chunks = {"visual": visual_boundaries, "text": text_boundaries}

        return DCMTOutput(
            logits=logits,
            chunks=chunks,
            alignment=token_alignment,
            loss=loss,
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "DCMTModel":
        """Load a pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        config = DCMTConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str):
        """Save model to checkpoint."""
        torch.save({
            "config": vars(self.config),
            "model_state_dict": self.state_dict(),
        }, path)
