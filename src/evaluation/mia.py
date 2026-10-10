"""Membership Inference Attack (MIA) evaluation for Privara (Stretch Goal).

Evaluates empirical privacy leakage between non-private and differentially private models
using shadow loss and confidence thresholding across member and non-member sets.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


class MembershipInferenceEvaluator:
    """Evaluates vulnerability to Membership Inference Attacks (MIA).
    
    Attacker Assumption:
    - Black-box access to model output probabilities or prediction losses.
    - Balanced member dataset (used in client training) and non-member dataset (unseen test samples).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def evaluate_model(
        self,
        model: nn.Module,
        member_dataset: Dataset,
        non_member_dataset: Dataset,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """Compute MIA metrics (AUC-ROC, attack accuracy, score distributions).
        
        Returns:
            Dictionary with attack AUC, attack accuracy, mean member/non-member loss & confidence.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")

        member_loader = DataLoader(member_dataset, batch_size=batch_size, shuffle=False)
        non_member_loader = DataLoader(non_member_dataset, batch_size=batch_size, shuffle=False)

        member_losses = []
        member_confs = []
        with torch.no_grad():
            for data, target in member_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                losses = criterion(logits, target)
                probs = torch.softmax(logits, dim=1)
                confs, _ = torch.max(probs, dim=1)

                member_losses.extend(losses.cpu().numpy().tolist())
                member_confs.extend(confs.cpu().numpy().tolist())

        non_member_losses = []
        non_member_confs = []
        with torch.no_grad():
            for data, target in non_member_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                losses = criterion(logits, target)
                probs = torch.softmax(logits, dim=1)
                confs, _ = torch.max(probs, dim=1)

                non_member_losses.extend(losses.cpu().numpy().tolist())
                non_member_confs.extend(confs.cpu().numpy().tolist())

        # Equalize lengths for balanced attack evaluation
        min_len = min(len(member_losses), len(non_member_losses))
        y_true = np.array([1] * min_len + [0] * min_len)
        # Attacker decision score: higher confidence / lower loss indicates member
        # Invert loss so higher score = more likely member
        scores = np.concatenate([
            -np.array(member_losses[:min_len]),
            -np.array(non_member_losses[:min_len])
        ])

        auc = float(roc_auc_score(y_true, scores))
        # Optimal threshold accuracy
        sorted_scores = np.sort(scores)
        best_acc = 0.5
        for threshold in sorted_scores[::max(1, len(scores) // 50)]:
            preds = (scores >= threshold).astype(int)
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = float(acc)

        return {
            "attack_auc": round(auc, 4),
            "attack_accuracy": round(best_acc, 4),
            "member_samples": min_len,
            "non_member_samples": min_len,
            "member_mean_loss": round(float(np.mean(member_losses[:min_len])), 4),
            "non_member_mean_loss": round(float(np.mean(non_member_losses[:min_len])), 4),
            "member_mean_confidence": round(float(np.mean(member_confs[:min_len])), 4),
            "non_member_mean_confidence": round(float(np.mean(non_member_confs[:min_len])), 4),
            "empirical_leakage_mitigated": auc < 0.55,
            "disclaimer": (
                "Empirical Membership Inference evaluation provides empirical sanity checking. "
                "A low empirical attack score does not replace the formal mathematical (ε, δ) "
                "guarantee provided by differential privacy."
            )
        }
