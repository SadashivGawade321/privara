"""Binary serializer and payload size measurement for Privara.

Measures actual on-the-wire byte sizes of uncompressed and compressed
model updates, including headers, indices, values, and transport metadata.
"""

from typing import Dict, Any, Tuple
import pickle
import io
import torch


class PayloadSerializer:
    """Serializes model updates and measures actual transmitted payload sizes in bytes."""

    @staticmethod
    def serialize_uncompressed(
        delta: Dict[str, torch.Tensor],
        metadata: Dict[str, Any]
    ) -> Tuple[bytes, int]:
        """Serialize full dense update with metadata and return (bytes, size_in_bytes)."""
        # Convert tensors to CPU numpy/float32 for standardized transport
        payload = {
            "type": "dense_delta",
            "metadata": metadata,
            "weights": {k: v.detach().cpu().numpy() for k, v in delta.items()}
        }
        buffer = io.BytesIO()
        pickle.dump(payload, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        raw_bytes = buffer.getvalue()
        return raw_bytes, len(raw_bytes)

    @staticmethod
    def serialize_compressed(
        compressed_dict: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Tuple[bytes, int]:
        """Serialize Top-K sparse update with metadata and return (bytes, size_in_bytes)."""
        payload = {
            "type": "topk_sparse_delta",
            "metadata": metadata,
            "compression_data": compressed_dict
        }
        buffer = io.BytesIO()
        pickle.dump(payload, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        raw_bytes = buffer.getvalue()
        return raw_bytes, len(raw_bytes)

    @staticmethod
    def serialize_global_model(
        model_state_dict: Dict[str, torch.Tensor],
        round_num: int
    ) -> Tuple[bytes, int]:
        """Serialize server-to-client global model broadcast and return (bytes, size_in_bytes)."""
        payload = {
            "type": "global_model_broadcast",
            "round": round_num,
            "weights": {k: v.detach().cpu().numpy() for k, v in model_state_dict.items()}
        }
        buffer = io.BytesIO()
        pickle.dump(payload, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        raw_bytes = buffer.getvalue()
        return raw_bytes, len(raw_bytes)

    @staticmethod
    def deserialize(raw_bytes: bytes) -> Dict[str, Any]:
        """Deserialize raw byte payload."""
        return pickle.loads(raw_bytes)
