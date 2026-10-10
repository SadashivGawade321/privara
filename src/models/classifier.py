"""Model architectures for Privara federated learning.

Features DP-compatible CNN and MLP architectures designed for CPU execution
and full compatibility with Opacus DP-SGD per-sample gradient clipping.
"""

from typing import Dict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.validators import ModuleValidator


class PrivaraCNN(nn.Module):
    """Compact Convolutional Neural Network compatible with Opacus DP-SGD.
    
    Uses GroupNorm instead of BatchNorm to preserve per-sample gradient independence.
    Input: (B, 1, 28, 28) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PrivaraMLP(nn.Module):
    """Compact Multi-Layer Perceptron for ultra-rapid CPU benchmarking."""
    def __init__(self, num_classes: int = 10, in_features: int = 784, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(architecture: str = "cnn", num_classes: int = 10) -> nn.Module:
    """Factory method returning an Opacus-validated PyTorch model."""
    arch = architecture.lower()
    if arch == "cnn":
        model = PrivaraCNN(num_classes=num_classes)
    elif arch == "mlp":
        model = PrivaraMLP(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Supported: ['cnn', 'mlp']")

    # Ensure model is strictly valid for Opacus DP-SGD
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_delta(
    global_model: nn.Module,
    local_model: nn.Module
) -> Dict[str, torch.Tensor]:
    """Compute parameter delta: delta = local - global."""
    global_params = dict(global_model.named_parameters())
    local_params = dict(local_model.named_parameters())
    delta = {}
    for name, p_local in local_params.items():
        if name in global_params:
            delta[name] = (p_local.data - global_params[name].data).clone().detach()
    return delta


def apply_model_delta(
    base_model: nn.Module,
    delta: Dict[str, torch.Tensor],
    weight: float = 1.0
) -> None:
    """Apply parameter delta in-place: base_param += weight * delta."""
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name in delta:
                param.data.add_(delta[name], alpha=weight)
