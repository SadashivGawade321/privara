"""Tests for dataset loading and client partitioning."""

import pytest
import torch
from src.data.dataset_loader import SyntheticDataset, load_dataset
from src.data.partitioner import DataPartitioner


def test_synthetic_dataset():
    ds = SyntheticDataset(num_samples=200, num_classes=10, seed=42)
    assert len(ds) == 200
    img, label = ds[0]
    assert img.shape == (1, 28, 28)
    assert 0 <= label.item() < 10


def test_partitioner_iid():
    ds = SyntheticDataset(num_samples=500, num_classes=10, seed=42)
    part = DataPartitioner(dataset=ds, num_clients=10, strategy="iid", seed=42)
    stats = part.get_client_stats()
    assert len(stats) == 10
    total_assigned = sum(s["num_samples"] for s in stats.values())
    assert total_assigned == 500
    for cid in [f"C{i+1}" for i in range(10)]:
        subset = part.get_client_subset(cid)
        assert len(subset) == 50


def test_partitioner_dirichlet_non_iid():
    ds = SyntheticDataset(num_samples=1000, num_classes=10, seed=42)
    part = DataPartitioner(
        dataset=ds,
        num_clients=10,
        strategy="dirichlet",
        dirichlet_alpha=0.2,
        min_samples_per_client=30,
        seed=42
    )
    stats = part.get_client_stats()
    assert len(stats) == 10
    for cid, s in stats.items():
        assert s["num_samples"] >= 30
        assert len(s["class_distribution"]) > 0


def test_partitioner_label_skew():
    ds = SyntheticDataset(num_samples=600, num_classes=10, seed=42)
    part = DataPartitioner(
        dataset=ds,
        num_clients=10,
        strategy="label_skew",
        classes_per_client=2,
        seed=42
    )
    stats = part.get_client_stats()
    assert len(stats) == 10
    for cid, s in stats.items():
        assert s["num_samples"] > 0
