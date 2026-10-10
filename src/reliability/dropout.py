"""Client dropout simulation for Privara.

Simulates deterministic client dropout failures without crashing the pipeline,
implementing a minimum-participant safeguard policy.
"""

from typing import List, Tuple, Dict
import numpy as np
from utils.logging import log_event


class DropoutSimulator:
    """Simulates client disconnections and network dropouts."""

    def __init__(self, dropout_prob: float = 0.1, min_participants: int = 2, seed: int = 42):
        self.dropout_prob = max(0.0, min(1.0, dropout_prob))
        self.min_participants = min_participants
        self.rng = np.random.RandomState(seed)

    def filter_active_clients(
        self,
        candidate_clients: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Determine which clients drop out and which remain active.
        
        Returns:
            (active_clients, dropped_clients)
        """
        if self.dropout_prob <= 0.0 or len(candidate_clients) <= self.min_participants:
            return list(candidate_clients), []

        active = []
        dropped = []

        for cid in candidate_clients:
            if self.rng.rand() < self.dropout_prob:
                dropped.append(cid)
            else:
                active.append(cid)

        # Safeguard: ensure we meet minimum participant policy
        if len(active) < self.min_participants:
            needed = self.min_participants - len(active)
            rescued = dropped[:needed]
            dropped = dropped[needed:]
            active.extend(rescued)

        if dropped:
            log_event(
                "CLIENT_DROPOUT",
                f"{len(dropped)} client(s) dropped out: {', '.join(dropped)}",
                level="warning"
            )

        return active, dropped
