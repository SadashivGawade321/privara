"""Top-K Gradient and Model-Update Sparsification for Privara.

Implements Top-K compression, bitmask/sparse serialization representation,
decompression/reconstruction, and relative reconstruction error measurement.
"""

from typing import Dict, Any, Tuple
import torch
import numpy as np


class TopKCompressor:
    """Compresses model update deltas using Top-K magnitude sparsification."""

    def __init__(self, topk_ratio: float = 0.5):
        """
        Args:
            topk_ratio: Fraction of largest magnitude values to retain (0.0 < ratio <= 1.0).
        """
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        self.topk_ratio = topk_ratio

    def compress(
        self,
        delta: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compress model update dictionary using Top-K sparsification.
        
        Uses bitmask compression for high bandwidth efficiency:
        Stores 1 bit per weight for presence mask + 32-bit floats for Top-K values.
        
        Returns:
            (compressed_payload, compression_stats)
        """
        compressed_payload: Dict[str, Any] = {
            "topk_ratio": self.topk_ratio,
            "layers": {}
        }
        
        total_elements = 0
        retained_elements = 0
        sum_orig_sq = 0.0
        sum_diff_sq = 0.0

        for name, tensor in delta.items():
            t_flat = tensor.detach().cpu().view(-1)
            numel = t_flat.numel()
            total_elements += numel

            k = max(1, int(round(numel * self.topk_ratio)))
            retained_elements += k

            # Get Top-K indices by absolute magnitude
            abs_flat = torch.abs(t_flat)
            values, indices = torch.topk(abs_flat, k=k, sorted=False)
            
            # Extract signed values at top indices
            signed_values = t_flat[indices]

            # Measure reconstruction error for this layer
            recon_flat = torch.zeros_like(t_flat)
            recon_flat[indices] = signed_values
            diff = t_flat - recon_flat

            sum_orig_sq += torch.sum(t_flat ** 2).item()
            sum_diff_sq += torch.sum(diff ** 2).item()

            # Pack boolean mask (1 bit per weight)
            mask = np.zeros(numel, dtype=bool)
            mask[indices.numpy()] = True
            packed_mask = np.packbits(mask)

            compressed_payload["layers"][name] = {
                "shape": list(tensor.shape),
                "packed_mask": packed_mask,
                "values": signed_values.to(torch.float32).numpy(),
                "indices": indices.to(torch.int32).numpy()
            }

        norm_orig = np.sqrt(sum_orig_sq)
        norm_diff = np.sqrt(sum_diff_sq)
        recon_error = float(norm_diff / (norm_orig + 1e-8))

        stats = {
            "topk_ratio": self.topk_ratio,
            "total_elements": total_elements,
            "retained_elements": retained_elements,
            "reconstruction_error": recon_error,
            "theoretical_sparsity": 1.0 - (retained_elements / max(1, total_elements))
        }

        return compressed_payload, stats

    def decompress(
        self,
        compressed_payload: Dict[str, Any],
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """Deconstruct compressed payload into reconstructed dense delta tensors."""
        reconstructed: Dict[str, torch.Tensor] = {}

        for name, layer_data in compressed_payload["layers"].items():
            shape = tuple(layer_data["shape"])
            numel = int(np.prod(shape))

            # Use indices + scatter for exact positional placement (avoids mask ordering issues)
            indices = torch.from_numpy(layer_data["indices"]).long()
            values = torch.from_numpy(layer_data["values"]).float()
            flat = torch.zeros(numel, dtype=torch.float32)
            flat.scatter_(0, indices, values)
            reconstructed[name] = flat.view(shape).to(device)

        return reconstructed

    @staticmethod
    def get_privacy_postprocessing_note() -> str:
        """Formal privacy remark on compression interaction with DP-SGD."""
        return (
            "Post-Processing Invariance: Top-K sparsification is applied strictly after "
            "DP-SGD gradient clipping and Gaussian noise perturbation. Under the post-processing "
            "property of differential privacy, any data-independent mapping f(M(x)) inherits "
            "the identical (ε, δ)-DP guarantee without privacy loss."
        )
