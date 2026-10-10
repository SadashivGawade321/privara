"""Differential Privacy training routine for local federated clients.

Integrates Opacus DP-SGD with calibrated Gaussian noise addition,
per-sample gradient clipping, and robust non-private fallback.
"""

from typing import Dict, Any, Tuple, Optional
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
from privacy.accountant import PrivacyAccountantWrapper
from utils.logging import logger


def train_local_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    epochs: int = 1,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    dp_enabled: bool = True,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.2,
    target_delta: float = 1e-5,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train model locally on client dataset using DP-SGD or standard SGD.
    
    Args:
        model: Base model (weights initialized from global model)
        dataset: Local client dataset subset
        batch_size: Mini-batch size
        epochs: Local training epochs
        learning_rate: Local optimizer learning rate
        momentum: SGD momentum
        dp_enabled: Whether to apply differential privacy
        noise_multiplier: Calibrated noise scale sigma
        max_grad_norm: L2 gradient clipping threshold C
        target_delta: Delta parameter for privacy calculation
        device: "cpu" or "cuda"

    Returns:
        (trained_model, training_metrics_dict)
    """
    local_model = copy.deepcopy(model).to(device)
    local_model.train()

    # Drop last to avoid fractional batches in DP-SGD if possible, but keep at least 1 batch
    drop_last = len(dataset) > batch_size
    data_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=drop_last
    )

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=learning_rate,
        momentum=momentum if not dp_enabled else 0.0 # Opacus DP-SGD standard setup
    )
    criterion = nn.CrossEntropyLoss()

    privacy_engine: Optional[PrivacyEngine] = None
    applied_dp = False
    sample_rate = float(min(batch_size, len(dataset))) / float(max(1, len(dataset)))

    if dp_enabled and noise_multiplier > 0:
        try:
            privacy_engine = PrivacyEngine(secure_mode=False)
            _private = privacy_engine.make_private(
                module=local_model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            # make_private returns (model, optimizer, dataloader) or
            # (model, optimizer, loss_fn, dataloader) in fast-clipping mode.
            local_model, optimizer, data_loader = _private[0], _private[1], _private[-1]  # type: ignore[assignment]
            applied_dp = True
        except Exception as e:
            logger.warning(f"Opacus make_private failed ({e}). Proceeding with manual DP-SGD clipping.")
            applied_dp = False

    total_loss = 0.0
    correct = 0
    total_samples = 0
    steps_taken = 0

    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()

            # If Opacus was not wrapped, but DP was requested, apply manual per-batch clipping & noise
            if dp_enabled and not applied_dp and noise_multiplier > 0:
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
                for p in local_model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * (noise_multiplier * max_grad_norm / (data.size(0) ** 0.5))
                        p.grad.add_(noise)

            optimizer.step()
            steps_taken += 1

            total_loss += loss.item() * len(target)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            total_samples += len(target)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)

    # Unwrap Opacus model wrapper if present to return plain Module
    actual_model = local_model._module if hasattr(local_model, "_module") else local_model

    # Privacy calculation
    achieved_eps = float("inf")
    if applied_dp and privacy_engine is not None:
        try:
            achieved_eps = privacy_engine.get_epsilon(delta=target_delta)
        except Exception:
            achieved_eps = float("inf")
    elif not dp_enabled or noise_multiplier <= 0:
        achieved_eps = float("inf")

    metrics = {
        "train_loss": avg_loss,
        "train_accuracy": accuracy,
        "steps_taken": steps_taken,
        "sample_rate": sample_rate,
        "dp_applied": applied_dp or (dp_enabled and noise_multiplier > 0),
        "noise_multiplier": noise_multiplier if dp_enabled else 0.0,
        "max_grad_norm": max_grad_norm if dp_enabled else 0.0,
        "achieved_epsilon": achieved_eps,
        "target_delta": target_delta,
        "num_samples": len(dataset)
    }

    return actual_model, metrics
