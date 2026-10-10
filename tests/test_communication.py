"""Tests for binary payload serialization and communication budget enforcement."""

import pytest
import torch
from src.communication.serializer import PayloadSerializer
from src.communication.budget import CommunicationBudgetManager
from src.communication.metrics import CommunicationMetricsTracker


def test_payload_serializer_measurement():
    delta = {"fc.weight": torch.randn(10, 10)}
    metadata = {"client_id": "C1", "round": 1}
    
    raw_bytes, size = PayloadSerializer.serialize_uncompressed(delta, metadata)
    assert size > 0
    assert len(raw_bytes) == size
    
    deserialized = PayloadSerializer.deserialize(raw_bytes)
    assert deserialized["metadata"]["client_id"] == "C1"
    assert "fc.weight" in deserialized["weights"]


def test_communication_budget_enforcement():
    # Set a small budget: 0.001 MB = ~1048 bytes
    mgr = CommunicationBudgetManager(budget_per_round_mb=0.001, budget_mode="upload", enforce_policy="reject")
    mgr.start_round(1)

    # First update: 500 bytes -> Accepted
    accepted1, _ = mgr.check_and_record_upload("C1", payload_bytes=500)
    assert accepted1 is True
    assert mgr.accepted_updates == 1

    # Second update: 700 bytes -> Total 1200 > 1048 -> Rejected!
    accepted2, reason = mgr.check_and_record_upload("C2", payload_bytes=700)
    assert accepted2 is False
    assert mgr.rejected_updates == 1
    assert "C2" in mgr.rejected_clients
    assert mgr.budget_exceeded is True

    summary = mgr.end_round()
    assert summary["rejected_updates"] == 1
    assert summary["accepted_updates"] == 1


def test_communication_metrics_tracker():
    tracker = CommunicationMetricsTracker()
    tracker.record_round(
        round_num=1,
        uncompressed_bytes=1000,
        compressed_bytes=400,
        metadata_bytes=50,
        overhead_bytes=30,
        time_seconds=1.2,
        compression_ratio=0.6,
        reconstruction_error=0.05
    )
    summary = tracker.get_summary()
    assert summary["overall_compression_ratio"] == 0.6
