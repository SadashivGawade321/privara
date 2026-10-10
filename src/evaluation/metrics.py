"""Evaluation metrics for Privara federated learning."""

from typing import Dict, Any, List
import numpy as np


def compute_accuracy_summary(predictions: List[int], targets: List[int]) -> Dict[str, float]:
    """Compute basic classification accuracy and error rates."""
    y_pred = np.array(predictions)
    y_true = np.array(targets)
    acc = np.mean(y_pred == y_true)
    return {
        "accuracy": float(acc),
        "error_rate": float(1.0 - acc),
        "total_samples": len(targets)
    }


def format_bytes_to_mb(byte_count: int) -> float:
    """Format raw bytes to Megabytes (MB)."""
    return round(byte_count / (1024 * 1024), 3)


def calculate_tradeoff_score(accuracy: float, epsilon: float, compression_ratio: float) -> float:
    """Composite utility-privacy-efficiency score for comparative rankings.
    
    Higher score indicates favorable balance between high accuracy, small epsilon, and high compression.
    """
    if epsilon == float("inf") or epsilon <= 0:
        eff_eps = 15.0  # Penalty for non-private baseline
    else:
        eff_eps = epsilon
    # Trade-off formula: Accuracy * (1 + Compression) / log2(1 + Epsilon)
    score = (accuracy * (1.0 + compression_ratio)) / np.log2(1.0 + eff_eps)
    return round(float(score), 3)
