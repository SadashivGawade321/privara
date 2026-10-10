"""Federated Client implementation for Privara.

Executes local model training, DP-SGD with privacy tracking,
optional Top-K gradient compression, and strict isolation of local data.
"""

from typing import Dict, Any, Tuple, Optional
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from src.privacy.dp_training import train_local_model
from src.models.classifier import compute_model_delta
from src.compression.topk import TopKCompressor
from src.communication.serializer import PayloadSerializer


class FederatedClient:
    """Simulated federated learning client."""

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        client_stats: Dict[str, Any],
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.stats = client_stats
        self.device = device
        self.status = "Active"  # "Active", "Delayed", "Dropped", "Rejected"
        self.last_update_round = 0
        self.last_accuracy = 0.0
        self.last_loss = 0.0

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def train_round(
        self,
        global_model: nn.Module,
        round_num: int,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        dp_enabled: bool = True,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.2,
        target_delta: float = 1e-5,
        compression_enabled: bool = True,
        topk_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """Perform one local training round and prepare serialized update package.
        
        Raw training data NEVER leaves this client instance.
        """
        # Train locally
        trained_local_model, train_metrics = train_local_model(
            model=global_model,
            dataset=self.dataset,
            batch_size=batch_size,
            epochs=local_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            dp_enabled=dp_enabled,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            target_delta=target_delta,
            device=self.device
        )

        # Compute parameter update delta: Delta = theta_local - theta_global
        delta = compute_model_delta(global_model, trained_local_model)

        metadata = {
            "client_id": self.client_id,
            "round": round_num,
            "num_samples": self.num_samples,
            "train_loss": train_metrics["train_loss"],
            "train_accuracy": train_metrics["train_accuracy"]
        }

        # Measure baseline uncompressed serialized size
        _, uncompressed_bytes = PayloadSerializer.serialize_uncompressed(delta, metadata)

        compressed_payload = None
        reconstruction_error = 0.0
        compressed_bytes = uncompressed_bytes

        if compression_enabled:
            compressor = TopKCompressor(topk_ratio=topk_ratio)
            compressed_dict, comp_stats = compressor.compress(delta)
            _, compressed_bytes = PayloadSerializer.serialize_compressed(compressed_dict, metadata)
            compressed_payload = compressed_dict
            reconstruction_error = comp_stats["reconstruction_error"]

        self.last_update_round = round_num
        self.last_accuracy = train_metrics["train_accuracy"]
        self.last_loss = train_metrics["train_loss"]

        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "raw_delta": delta if not compression_enabled else None,
            "compressed_payload": compressed_payload,
            "is_compressed": compression_enabled,
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": compressed_bytes,
            "reconstruction_error": reconstruction_error,
            "train_metrics": train_metrics
        }
