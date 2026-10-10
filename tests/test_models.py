"""Tests for model architectures and delta calculations."""

import pytest
import torch
from opacus.validators import ModuleValidator
from src.models.classifier import get_model, count_parameters, compute_model_delta, apply_model_delta


def test_cnn_architecture_and_opacus_compatibility():
    model = get_model("cnn", num_classes=10)
    assert ModuleValidator.is_valid(model)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)
    assert count_parameters(model) > 1000


def test_mlp_architecture_and_opacus_compatibility():
    model = get_model("mlp", num_classes=10)
    assert ModuleValidator.is_valid(model)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_model_delta_computation_and_application():
    base_model = get_model("mlp", num_classes=10)
    local_model = get_model("mlp", num_classes=10)
    local_model.load_state_dict(base_model.state_dict())
    
    # Mutate local model
    with torch.no_grad():
        for p in local_model.parameters():
            p.add_(1.0)

    delta = compute_model_delta(base_model, local_model)
    assert len(delta) > 0
    for name, d_tensor in delta.items():
        assert torch.allclose(d_tensor, torch.ones_like(d_tensor), atol=1e-4)

    # Apply delta to base model
    apply_model_delta(base_model, delta, weight=1.0)
    for p_base, p_local in zip(base_model.parameters(), local_model.parameters()):
        assert torch.allclose(p_base, p_local, atol=1e-4)
