"""Tests for Membership Inference Attack (MIA) evaluation."""

import pytest
import torch
from src.data.dataset_loader import SyntheticDataset
from src.models.classifier import get_model
from src.evaluation.mia import MembershipInferenceEvaluator


def test_membership_inference_evaluator():
    member_data = SyntheticDataset(num_samples=100, num_classes=10, seed=1)
    non_member_data = SyntheticDataset(num_samples=100, num_classes=10, seed=2)
    model = get_model("mlp", num_classes=10)

    evaluator = MembershipInferenceEvaluator()
    result = evaluator.evaluate_model(
        model=model,
        member_dataset=member_data,
        non_member_dataset=non_member_data,
        batch_size=32
    )

    assert "attack_auc" in result
    assert "attack_accuracy" in result
    assert 0.0 <= result["attack_auc"] <= 1.0
    assert 0.0 <= result["attack_accuracy"] <= 1.0
    assert result["member_samples"] == 100
