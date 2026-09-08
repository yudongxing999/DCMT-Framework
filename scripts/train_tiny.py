#!/usr/bin/env python3
"""Tiny CPU-friendly DCMT training with supervised token-level alignment.

Loss = CE(labels) + w_contrast * contrastive + w_boundary * BoundaryLoss
     + w_token_align * token_InfoNCE(alignment, pairs)

Default out: results/tiny_align_v2, epochs=8.
"""
