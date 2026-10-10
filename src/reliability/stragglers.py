"""Client straggler and latency delay simulation for Privara.

Simulates delayed client responses deterministically for federated robustness evaluation.
"""

from typing import List, Tuple, Dict
import time
import numpy as np
from utils.logging import log_event


class StragglerSimulator:
    """Simulates stragglers (heterogeneous response latency)."""

    def __init__(
        self,
        straggler_prob: float = 0.1,
        simulated_delay_sec: float = 0.05,
        seed: int = 42
    ):
        self.straggler_prob = max(0.0, min(1.0, straggler_prob))
        self.simulated_delay_sec = simulated_delay_sec
        self.rng = np.random.RandomState(seed + 100)

    def evaluate_stragglers(
        self,
        active_clients: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Identify straggler clients among active participants.
        
        Returns:
            (on_time_clients, delayed_clients)
        """
        if self.straggler_prob <= 0.0:
            return list(active_clients), []

        on_time = []
        delayed = []

        for cid in active_clients:
            if self.rng.rand() < self.straggler_prob:
                delayed.append(cid)
            else:
                on_time.append(cid)

        if delayed:
            log_event(
                "CLIENT_STRAGGLER",
                f"{len(delayed)} client(s) delayed/straggling: {', '.join(delayed)}",
                level="info"
            )

        return on_time, delayed

    def simulate_delay(self, is_straggler: bool) -> float:
        """Emulate minor deterministic network latency."""
        if is_straggler and self.simulated_delay_sec > 0:
            time.sleep(self.simulated_delay_sec)
            return self.simulated_delay_sec
        return 0.0
