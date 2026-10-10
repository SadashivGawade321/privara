"""Tests for Top-K gradient compression and reconstruction."""

import pytest
import torch
from src.compression.topk import TopKCompressor


def test_topk_compression_and_decompression():
    compressor = TopKCompressor(topk_ratio=0.5)
    
    # Create synthetic delta
    delta = {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10)
    }

    compressed_dict, stats = compressor.compress(delta)
    assert stats["topk_ratio"] == 0.5
    assert stats["total_elements"] == 110
    assert stats["retained_elements"] == 55
    assert 0.0 <= stats["reconstruction_error"] <= 1.0

    # Decompress
    recon = compressor.decompress(compressed_dict)
    assert "layer1.weight" in recon
    assert "layer1.bias" in recon
    assert recon["layer1.weight"].shape == (10, 10)
    assert recon["layer1.bias"].shape == (10,)

    # Verify sparsity: half the values in reconstruction must be zero
    zero_count = torch.sum(recon["layer1.weight"] == 0.0).item()
    assert zero_count == 50


def test_topk_boundary_ratios():
    c_full = TopKCompressor(topk_ratio=1.0)
    delta = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    comp_dict, stats = c_full.compress(delta)
    assert stats["reconstruction_error"] == 0.0
    recon = c_full.decompress(comp_dict)
    assert torch.allclose(recon["w"], delta["w"])
