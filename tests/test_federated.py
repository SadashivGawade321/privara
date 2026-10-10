"""End-to-end federated simulation tests for Privara."""

import pytest
import yaml
from src.data.dataset_loader import SyntheticDataset
from src.federated.simulator import FederatedSimulator


def test_federated_simulator_single_round():
    config = {
        "dataset": {"name": "Synthetic"},
        "partitioning": {
            "num_clients": 10,
            "strategy": "dirichlet",
            "dirichlet_alpha": 0.5,
            "min_samples_per_client": 30
        },
        "model": {"architecture": "mlp", "num_classes": 10},
        "federated": {
            "num_rounds": 1,
            "client_fraction": 1.0,
            "local_epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "min_participants": 2
        },
        "privacy": {
            "enabled": True,
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.2,
            "target_delta": 1e-5,
            "accountant": "rdp",
            "target_epsilon": 5.0
        },
        "compression": {
            "enabled": True,
            "topk_ratio": 0.5
        },
        "communication": {
            "budget_per_round_mb": 50.0,
            "budget_mode": "bidirectional",
            "enforce_policy": "reject"
        },
        "reliability": {
            "dropout_prob": 0.1,
            "straggler_prob": 0.1,
            "simulated_delay_sec": 0.0
        },
        "evaluation": {
            "test_batch_size": 32
        },
        "reproducibility": {
            "seed": 42
        }
    }

    train_data = SyntheticDataset(num_samples=500, num_classes=10, seed=42)
    test_data = SyntheticDataset(num_samples=100, num_classes=10, seed=99)

    sim = FederatedSimulator(config, train_dataset=train_data, test_dataset=test_data)
    summary = sim.run_simulation(num_rounds=1)

    assert summary["rounds_completed"] == 1
    assert 0.0 <= summary["final_test_accuracy"] <= 1.0
    assert len(summary["round_history"]) == 1

    round_1 = summary["round_history"][0]
    assert round_1["total_clients"] == 10
    assert round_1["accepted_clients_count"] >= 2
    assert round_1["achieved_epsilon"] > 0
    assert round_1["compressed_mb"] > 0
    # Allow 50% overhead for pickle serialization (small models may see overhead)
    assert round_1["uncompressed_mb"] * 1.5 >= round_1["compressed_mb"]
