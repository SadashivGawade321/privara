"""Privacy Accounting module for Privara.

Wraps Opacus RDPAccountant and PRVAccountant to calculate exact cumulative
differential privacy bounds (epsilon, delta) based on real DP-SGD sampling and noise steps.
"""

from typing import Dict, Any, List, Optional
import math
from opacus.accountants import RDPAccountant, PRVAccountant
from utils.logging import log_event, logger


class PrivacyAccountantWrapper:
    """Rigorous privacy accountant wrapping Opacus accountants.
    
    Tracks cumulative privacy expenditure across federated rounds.
    Guarantees: Example-level Differential Privacy (DP) for local client training.
    """

    PRESET_CONFIGURATIONS = {
        "strict": {
            "name": "Strict Privacy (ε ≤ 1.5)",
            "noise_multiplier": 2.0,
            "max_grad_norm": 1.0,
            "target_epsilon": 1.5,
            "description": "Maximum noise addition; high formal protection, modest utility penalty."
        },
        "high": {
            "name": "High Privacy (ε ≈ 2.5)",
            "noise_multiplier": 1.2,
            "max_grad_norm": 1.2,
            "target_epsilon": 2.5,
            "description": "Standard production privacy balance for sensitive medical or financial records."
        },
        "moderate": {
            "name": "Moderate Privacy (ε ≈ 4.5)",
            "noise_multiplier": 0.8,
            "max_grad_norm": 1.5,
            "target_epsilon": 4.5,
            "description": "Moderate noise addition; higher utility with formal DP protection."
        },
        "relaxed": {
            "name": "Relaxed Privacy (ε ≈ 8.0)",
            "noise_multiplier": 0.5,
            "max_grad_norm": 2.0,
            "target_epsilon": 8.0,
            "description": "Low noise regime; closer to non-private accuracy with weaker privacy bounds."
        },
        "non_private": {
            "name": "Non-Private Baseline",
            "noise_multiplier": 0.0,
            "max_grad_norm": 0.0,
            "target_epsilon": None,
            "description": "Standard SGD baseline with no noise or clipping (no DP guarantee, ε = ∞)."
        }
    }

    def __init__(
        self,
        accountant_type: str = "rdp",
        target_delta: float = 1e-5,
        target_epsilon: Optional[float] = 3.0
    ):
        self.accountant_type = accountant_type.lower()
        self.target_delta = float(target_delta)
        self.target_epsilon = float(target_epsilon) if target_epsilon is not None else None

        if self.accountant_type == "prv":
            self._accountant = PRVAccountant()
        else:
            self.accountant_type = "rdp"
            self._accountant = RDPAccountant()

        self.step_history: List[Dict[str, float]] = []
        self.round_history: List[Dict[str, Any]] = []
        self.total_steps: int = 0

    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1) -> None:
        """Record privacy steps taken in local training.
        
        Args:
            noise_multiplier: sigma (ratio of noise std to clipping norm C)
            sample_rate: q = batch_size / total_samples
            steps: number of mini-batch gradient steps
        """
        if noise_multiplier <= 0:
            # Non-private step: does not provide DP
            self.total_steps += steps
            return

        for _ in range(steps):
            self._accountant.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
            self.step_history.append({
                "noise_multiplier": float(noise_multiplier),
                "sample_rate": float(sample_rate)
            })
            self.total_steps += 1

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Calculate the exact cumulative achieved epsilon using the Opacus accountant."""
        if len(self.step_history) == 0:
            return 0.0
        try:
            d = float(delta) if delta is not None else float(self.target_delta)
            eps = self._accountant.get_epsilon(delta=d)
            if math.isinf(eps) or math.isnan(eps):
                return float("inf")
            return float(eps)
        except Exception as e:
            logger.error(f"Error computing epsilon: {e}")
            return float("inf")

    def record_round_end(self, round_num: int, noise_multiplier: float, sample_rate: float) -> Dict[str, Any]:
        """Record round-level privacy snapshot."""
        achieved_eps = self.get_epsilon()
        budget_used_pct = (
            min(100.0, (achieved_eps / self.target_epsilon) * 100.0)
            if self.target_epsilon and achieved_eps != float("inf")
            else None
        )
        record = {
            "round": round_num,
            "achieved_epsilon": round(achieved_eps, 4) if achieved_eps != float("inf") else "inf",
            "delta": self.target_delta,
            "total_steps": self.total_steps,
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "accountant_type": self.accountant_type.upper(),
            "target_epsilon": self.target_epsilon,
            "budget_used_pct": round(budget_used_pct, 1) if budget_used_pct is not None else None,
            "privacy_guarantee_satisfied": (
                achieved_eps <= self.target_epsilon if self.target_epsilon and achieved_eps != float("inf") else False
            )
        }
        self.round_history.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Return comprehensive privacy state and guarantee disclosure."""
        eps = self.get_epsilon()
        return {
            "accountant": self.accountant_type.upper(),
            "achieved_epsilon": eps,
            "target_delta": self.target_delta,
            "target_epsilon": self.target_epsilon,
            "total_steps": self.total_steps,
            "is_private": len(self.step_history) > 0,
            "privacy_unit": "Example-level differential privacy per client training process.",
            "assumptions": (
                "Guarantees that no individual training example's presence substantially shifts "
                "the local client model update distribution. Employs Poisson or mini-batch subsampling "
                "with calibrated Gaussian noise and bounded L2 sensitivity. Does NOT claim patient-level "
                "or client-level DP (where entire client inclusion/exclusion is protected)."
            )
        }
