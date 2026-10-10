"""Tests for Differential Privacy and Opacus accountant integration."""

import pytest
import torch
from src.privacy.accountant import PrivacyAccountantWrapper
from src.privacy.dp_training import train_local_model
from src.data.dataset_loader import SyntheticDataset
from src.models.classifier import get_model


def test_rdp_accountant_calculation():
    acc = PrivacyAccountantWrapper(accountant_type="rdp", target_delta=1e-5, target_epsilon=3.0)
    assert acc.get_epsilon() == 0.0
    acc.step(noise_multiplier=1.0, sample_rate=0.05, steps=5)
    eps = acc.get_epsilon()
    assert 0.0 < eps < 20.0
    
    # Step again and verify monotonic increase of epsilon
    acc.step(noise_multiplier=1.0, sample_rate=0.05, steps=5)
    eps2 = acc.get_epsilon()
    assert eps2 > eps


def test_prv_accountant_calculation():
    acc = PrivacyAccountantWrapper(accountant_type="prv", target_delta=1e-5, target_epsilon=3.0)
    acc.step(noise_multiplier=1.2, sample_rate=0.05, steps=4)
    eps = acc.get_epsilon()
    assert 0.0 < eps < 20.0


def test_local_dp_training_loop():
    dataset = SyntheticDataset(num_samples=64, num_classes=10, seed=42)
    model = get_model("mlp", num_classes=10)

    trained_model, metrics = train_local_model(
        model=model,
        dataset=dataset,
        batch_size=16,
        epochs=1,
        learning_rate=0.01,
        dp_enabled=True,
        noise_multiplier=1.0,
        max_grad_norm=1.2,
        target_delta=1e-5
    )

    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert metrics["dp_applied"] is True
    assert metrics["achieved_epsilon"] > 0
