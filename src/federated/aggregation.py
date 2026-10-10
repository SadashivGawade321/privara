"""Server-side aggregation algorithms for Privara federated learning.

Implements FedAvg (Federated Averaging) with sample-weighting,
sparse update reconstruction, and strict validation of accepted updates.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from compression.topk import TopKCompressor
from models.classifier import apply_model_delta
from utils.logging import log_event


def aggregate_updates_fedavg(
    global_model: nn.Module,
    client_packages: List[Dict[str, Any]],
    device: str = "cpu"
) -> Dict[str, Any]:
    """Aggregate client updates using weighted FedAvg.
    
    Delta_global = sum_{k in S} (n_k / sum_j n_j) * Delta_k
    theta^{t+1} = theta^t + Delta_global
    
    Args:
        global_model: Current global PyTorch model (updated in-place)
        client_packages: List of valid, accepted client packages
        device: "cpu" or "cuda"

    Returns:
        aggregation_metadata dictionary
    """
    if not client_packages:
        raise ValueError("Cannot aggregate empty client packages list.")

    total_samples = sum(pkg["num_samples"] for pkg in client_packages)
    if total_samples <= 0:
        raise ValueError("Total participating client sample count must be positive.")

    # Reconstruct deltas if compressed
    reconstructed_deltas: List[Tuple[Dict[str, torch.Tensor], int]] = []
    
    for pkg in client_packages:
        weight_n = pkg["num_samples"]
        if pkg["is_compressed"]:
            compressor = TopKCompressor(topk_ratio=pkg["compressed_payload"]["topk_ratio"])
            delta = compressor.decompress(pkg["compressed_payload"], device=device)
        else:
            delta = {k: v.to(device) for k, v in pkg["raw_delta"].items()}
        reconstructed_deltas.append((delta, weight_n))

    # Compute weighted FedAvg delta
    aggregated_delta: Dict[str, torch.Tensor] = {}
    base_delta, _ = reconstructed_deltas[0]
    for param_name in base_delta.keys():
        aggregated_delta[param_name] = torch.zeros_like(base_delta[param_name])

    for delta, n_samples in reconstructed_deltas:
        weight = float(n_samples) / float(total_samples)
        for param_name in delta.keys():
            aggregated_delta[param_name].add_(delta[param_name], alpha=weight)

    # Apply aggregated delta to global model: theta^{t+1} = theta^t + aggregated_delta
    apply_model_delta(global_model, aggregated_delta, weight=1.0)

    log_event(
        "FEDAVG_AGGREGATION",
        f"Aggregated {len(client_packages)} accepted updates ({total_samples} total samples)",
        details=f"Clients: {', '.join(pkg['client_id'] for pkg in client_packages)}"
    )

    return {
        "num_aggregated_clients": len(client_packages),
        "total_samples": total_samples,
        "clients": [pkg["client_id"] for pkg in client_packages]
    }
