"""Federated Server implementation for Privara.

Coordinates global model broadcasts, round evaluations on held-out test data,
budget enforcement, privacy accounting, and model aggregation.
"""

from typing import Dict, Any, Tuple, Optional, List
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.federated.aggregation import aggregate_updates_fedavg
from src.communication.serializer import PayloadSerializer
from src.communication.budget import CommunicationBudgetManager
from src.communication.metrics import CommunicationMetricsTracker
from src.privacy.accountant import PrivacyAccountantWrapper
from src.utils.logging import log_event, logger


class FederatedServer:
    """Central federated coordinator server."""

    def __init__(
        self,
        global_model: nn.Module,
        test_dataset: Dataset,
        privacy_accountant: PrivacyAccountantWrapper,
        budget_manager: CommunicationBudgetManager,
        metrics_tracker: CommunicationMetricsTracker,
        device: str = "cpu"
    ):
        self.global_model = global_model.to(device)
        self.test_dataset = test_dataset
        self.privacy_accountant = privacy_accountant
        self.budget_manager = budget_manager
        self.metrics_tracker = metrics_tracker
        self.device = device

        self.current_round: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_global_model(self, batch_size: int = 64) -> Tuple[float, float]:
        """Evaluate current global model on held-out test dataset.
        
        Returns:
            (test_accuracy, test_loss)
        """
        self.global_model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item() * len(target)
                preds = outputs.argmax(dim=1, keepdim=True)
                correct += preds.eq(target.view_as(preds)).sum().item()
                total_samples += len(target)

        test_acc = correct / max(1, total_samples)
        test_loss = total_loss / max(1, total_samples)
        return float(test_acc), float(test_loss)

    def prepare_broadcast_payload(self) -> Tuple[Dict[str, torch.Tensor], int]:
        """Prepare global model weights for client broadcast and measure download byte size."""
        state_dict = copy.deepcopy(self.global_model.state_dict())
        _, broadcast_bytes = PayloadSerializer.serialize_global_model(state_dict, self.current_round)
        return state_dict, broadcast_bytes

    def aggregate_round(
        self,
        accepted_packages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate accepted client packages into global model."""
        if len(accepted_packages) == 0:
            logger.warning(f"Round {self.current_round}: No client packages to aggregate!")
            return {"num_aggregated_clients": 0, "total_samples": 0}

        return aggregate_updates_fedavg(
            global_model=self.global_model,
            client_packages=accepted_packages,
            device=self.device
        )
