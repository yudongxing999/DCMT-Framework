"""CPU smoke test: randomly initialized tiny DCMT forward pass (no downloads)."""

import torch

from dcmt import DCMTModel, DCMTConfig


def test_forward_smoke_cpu():
    torch.manual_seed(0)
    config = DCMTConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=128,
        v_layers=1,
        patch_size=16,
        img_size=64,
        vocab_size=1000,
        max_seq_length=32,
        num_labels=10,
        dropout=0.0,
        boundary_threshold=0.5,
    )
    model = DCMTModel(config)
    model.eval()

    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    with torch.no_grad():
        out = model(
            images,
            input_ids,
            attention_mask=attention_mask,
            return_chunks=True,
            return_alignment=True,
        )

    assert out.logits.shape == (2, 10), f"unexpected logits shape {out.logits.shape}"
    assert isinstance(out.chunks, dict)
    assert "visual" in out.chunks and "text" in out.chunks
    assert out.chunks["visual"].shape[0] == 2
    assert out.chunks["text"].shape[0] == 2
    # Token-level alignment [B, N_v, N_t]
    assert out.alignment is not None
    assert out.alignment.ndim == 3
    assert out.alignment.shape[0] == 2
    n_v = out.chunks["visual"].shape[1]
    n_t = out.chunks["text"].shape[1]
    assert out.alignment.shape == (2, n_v, n_t)


def test_forward_with_labels_loss():
    torch.manual_seed(1)
    config = DCMTConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=128,
        v_layers=1,
        patch_size=16,
        img_size=64,
        vocab_size=500,
        max_seq_length=16,
        num_labels=5,
        dropout=0.0,
    )
    model = DCMTModel(config)
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 500, (2, 8))
    labels = torch.randint(0, 5, (2,))
    out = model(images, input_ids, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


if __name__ == "__main__":
    test_forward_smoke_cpu()
    test_forward_with_labels_loss()
    print("smoke ok")
