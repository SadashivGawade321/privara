"""Communication budget enforcement for Privara.

Monitors and strictly enforces per-round transmission caps (in bytes / MB)
for upload or bidirectional federated communication traffic.
"""

from typing import Dict, Any, List, Tuple
from utils.logging import log_event, logger


class CommunicationBudgetManager:
    """Manages and enforces communication budget per federated round."""

    def __init__(
        self,
        budget_per_round_mb: float = 20.0,
        budget_mode: str = "bidirectional",  # "bidirectional" or "upload"
        enforce_policy: str = "reject"       # "reject", "throttle", or "warn"
    ):
        self.budget_per_round_mb = budget_per_round_mb
        self.budget_bytes = int(budget_per_round_mb * 1024 * 1024)
        self.budget_mode = budget_mode.lower()
        self.enforce_policy = enforce_policy.lower()

        # Per-round state
        self.current_round: int = 0
        self.download_bytes: int = 0
        self.upload_bytes: int = 0
        self.accepted_updates: int = 0
        self.rejected_updates: int = 0
        self.rejected_clients: List[str] = []
        self.budget_exceeded: bool = False

        # Round history
        self.round_history: List[Dict[str, Any]] = []

    def start_round(self, round_num: int) -> None:
        """Reset per-round communication counters for a new round."""
        self.current_round = round_num
        self.download_bytes = 0
        self.upload_bytes = 0
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.rejected_clients = []
        self.budget_exceeded = False

    def record_download(self, bytes_count: int, num_clients: int) -> None:
        """Record server-to-client broadcast bytes (global model download)."""
        self.download_bytes += bytes_count * num_clients

    def check_and_record_upload(
        self,
        client_id: str,
        payload_bytes: int
    ) -> Tuple[bool, str]:
        """Check if client update fits within budget and record traffic.
        
        Returns:
            (is_accepted, reason_message)
        """
        current_tracked = (
            (self.download_bytes + self.upload_bytes + payload_bytes)
            if self.budget_mode == "bidirectional"
            else (self.upload_bytes + payload_bytes)
        )

        if current_tracked > self.budget_bytes:
            self.budget_exceeded = True
            if self.enforce_policy == "reject":
                self.rejected_updates += 1
                self.rejected_clients.append(client_id)
                reason = (
                    f"Rejected {client_id}: update ({payload_bytes / (1024*1024):.2f} MB) "
                    f"exceeds remaining budget ({self.budget_per_round_mb:.2f} MB limit)."
                )
                log_event("BUDGET_REJECT", reason, level="warning")
                return False, reason
            else:
                # Warning mode: allow but flag
                self.upload_bytes += payload_bytes
                self.accepted_updates += 1
                return True, "Warning: round budget exceeded but policy allows transmission."
        else:
            self.upload_bytes += payload_bytes
            self.accepted_updates += 1
            return True, "Accepted within budget."

    def end_round(self) -> Dict[str, Any]:
        """Finalize round communication metrics."""
        total_bytes = self.download_bytes + self.upload_bytes
        summary = {
            "round": self.current_round,
            "budget_mb": self.budget_per_round_mb,
            "budget_bytes": self.budget_bytes,
            "budget_mode": self.budget_mode,
            "download_bytes": self.download_bytes,
            "upload_bytes": self.upload_bytes,
            "total_bytes": total_bytes,
            "download_mb": round(self.download_bytes / (1024 * 1024), 3),
            "upload_mb": round(self.upload_bytes / (1024 * 1024), 3),
            "total_mb": round(total_bytes / (1024 * 1024), 3),
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "rejected_clients": list(self.rejected_clients),
            "budget_exceeded": self.budget_exceeded,
            "budget_utilization_pct": round((total_bytes / max(1, self.budget_bytes)) * 100.0, 1)
        }
        self.round_history.append(summary)
        return summary
