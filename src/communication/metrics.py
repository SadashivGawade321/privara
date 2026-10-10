"""Communication metrics tracker for Privara.

Records round-wise bytes, compression ratios, overhead breakdown,
and cumulative bandwidth savings.
"""

from typing import Dict, Any, List


class CommunicationMetricsTracker:
    """Tracks and analyzes communication traffic across federated rounds."""

    def __init__(self):
        self.round_metrics: List[Dict[str, Any]] = []
        self.total_uncompressed_bytes: int = 0
        self.total_compressed_bytes: int = 0

    def record_round(
        self,
        round_num: int,
        uncompressed_bytes: int,
        compressed_bytes: int,
        metadata_bytes: int,
        overhead_bytes: int,
        time_seconds: float,
        compression_ratio: float,
        reconstruction_error: float
    ) -> Dict[str, Any]:
        """Record communication statistics for a single round."""
        self.total_uncompressed_bytes += uncompressed_bytes
        self.total_compressed_bytes += compressed_bytes

        record = {
            "round": round_num,
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": compressed_bytes,
            "metadata_bytes": metadata_bytes,
            "overhead_bytes": overhead_bytes,
            "uncompressed_mb": round(uncompressed_bytes / (1024 * 1024), 3),
            "compressed_mb": round(compressed_bytes / (1024 * 1024), 3),
            "metadata_mb": round(metadata_bytes / (1024 * 1024), 3),
            "overhead_mb": round(overhead_bytes / (1024 * 1024), 3),
            "compression_ratio": round(compression_ratio, 3),
            "reconstruction_error": round(reconstruction_error, 4),
            "time_seconds": round(time_seconds, 2)
        }
        self.round_metrics.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregate communication summary across all recorded rounds."""
        if not self.round_metrics:
            return {
                "total_uncompressed_mb": 0.0,
                "total_compressed_mb": 0.0,
                "overall_compression_ratio": 0.0,
                "avg_uncompressed_mb": 0.0,
                "avg_compressed_mb": 0.0,
                "avg_round_time_sec": 0.0
            }

        uncomp_mb = self.total_uncompressed_bytes / (1024 * 1024)
        comp_mb = self.total_compressed_bytes / (1024 * 1024)
        savings_ratio = (
            1.0 - (comp_mb / max(1e-5, uncomp_mb))
            if uncomp_mb > 0
            else 0.0
        )
        avg_time = sum(r["time_seconds"] for r in self.round_metrics) / len(self.round_metrics)

        return {
            "total_uncompressed_mb": round(uncomp_mb, 2),
            "total_compressed_mb": round(comp_mb, 2),
            "overall_compression_ratio": round(savings_ratio, 3),
            "overall_savings_pct": round(savings_ratio * 100.0, 1),
            "avg_uncompressed_mb": round(uncomp_mb / len(self.round_metrics), 2),
            "avg_compressed_mb": round(comp_mb / len(self.round_metrics), 2),
            "avg_round_time_sec": round(avg_time, 2)
        }
