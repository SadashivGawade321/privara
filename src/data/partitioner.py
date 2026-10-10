"""Client data partitioner for Federated Learning.

Supports IID, Dirichlet Non-IID, and Label-Skew partitioning strategies.
Ensures zero test-set leakage, configurable heterogeneity parameter,
and comprehensive per-client statistical reporting.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from collections import Counter
from utils.logging import log_event


class DataPartitioner:
    """Partitions a training dataset among N federated clients."""

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int = 10,
        strategy: str = "dirichlet",
        dirichlet_alpha: float = 0.5,
        classes_per_client: int = 3,
        min_samples_per_client: int = 50,
        seed: int = 42
    ):
        self.dataset = dataset
        self.num_clients = num_clients
        self.strategy = strategy.lower()
        self.dirichlet_alpha = dirichlet_alpha
        self.classes_per_client = classes_per_client
        self.min_samples_per_client = min_samples_per_client
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Extract targets
        self.targets = self._extract_targets()
        self.num_classes = len(np.unique(self.targets))

        # Partition mapping: client_id (e.g. "C1") -> list of sample indices
        self.client_indices: Dict[str, List[int]] = {}
        self.client_subsets: Dict[str, Subset] = {}
        self.client_stats: Dict[str, Dict[str, Any]] = {}

        self._partition()

    def _extract_targets(self) -> np.ndarray:
        """Extract array of targets from dataset."""
        if hasattr(self.dataset, "tensors") and len(self.dataset.tensors) >= 2:
            return self.dataset.tensors[1].cpu().numpy()
        elif hasattr(self.dataset, "targets"):
            targets = self.dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets.cpu().numpy()
            return np.array(targets)
        elif hasattr(self.dataset, "labels"):
            labels = self.dataset.labels
            if isinstance(labels, torch.Tensor):
                return labels.cpu().numpy()
            return np.array(labels)
        else:
            # Fallback iteration
            targets = [self.dataset[i][1] for i in range(len(self.dataset))]
            if isinstance(targets[0], torch.Tensor):
                targets = [t.item() for t in targets]
            return np.array(targets)

    def _partition(self) -> None:
        """Execute partitioning based on configured strategy."""
        client_ids = [f"C{i+1}" for i in range(self.num_clients)]
        total_samples = len(self.targets)

        if self.strategy == "iid":
            self._partition_iid(client_ids)
        elif self.strategy == "dirichlet":
            self._partition_dirichlet(client_ids)
        elif self.strategy == "label_skew":
            self._partition_label_skew(client_ids)
        else:
            raise ValueError(f"Unknown partitioning strategy: {self.strategy}")

        # Compute client stats and subsets
        for cid, indices in self.client_indices.items():
            self.client_subsets[cid] = Subset(self.dataset, indices)
            labels = self.targets[indices]
            counts = Counter(labels)
            class_dist = {int(k): int(v) for k, v in sorted(counts.items())}
            present_classes = sorted(list(class_dist.keys()))
            
            self.client_stats[cid] = {
                "client_id": cid,
                "num_samples": len(indices),
                "classes": present_classes,
                "class_distribution": class_dist,
                "heterogeneity_alpha": self.dirichlet_alpha if self.strategy == "dirichlet" else None
            }

        log_event(
            "PARTITION_COMPLETE",
            f"Partitioned {total_samples} samples across {self.num_clients} clients ({self.strategy})",
            details=f"Strategy: {self.strategy}, Alpha: {self.dirichlet_alpha if self.strategy == 'dirichlet' else 'N/A'}"
        )

    def _partition_iid(self, client_ids: List[str]) -> None:
        """Uniformly random IID distribution."""
        indices = np.arange(len(self.targets))
        self.rng.shuffle(indices)
        splits = np.array_split(indices, self.num_clients)
        for i, cid in enumerate(client_ids):
            self.client_indices[cid] = splits[i].tolist()

    def _partition_dirichlet(self, client_ids: List[str]) -> None:
        """Dirichlet Non-IID label distribution partitioning."""
        num_samples = len(self.targets)
        client_idx_map = {cid: [] for cid in client_ids}

        # Organize sample indices by class
        class_indices = [
            np.where(self.targets == c)[0] for c in range(self.num_classes)
        ]
        for c in range(self.num_classes):
            self.rng.shuffle(class_indices[c])

        # For each class, sample proportions across clients from Dirichlet(alpha)
        for c in range(self.num_classes):
            c_indices = class_indices[c]
            if len(c_indices) == 0:
                continue

            proportions = self.rng.dirichlet(
                np.repeat(self.dirichlet_alpha, self.num_clients)
            )
            # Normalize to avoid rounding zero
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
            splits = np.split(c_indices, split_points)

            for i, cid in enumerate(client_ids):
                client_idx_map[cid].extend(splits[i].tolist())

        # Check minimum samples per client; rebalance if a client has too few
        for cid in client_ids:
            if len(client_idx_map[cid]) < self.min_samples_per_client:
                needed = self.min_samples_per_client - len(client_idx_map[cid])
                # Find richest client
                richest = max(client_ids, key=lambda c: len(client_idx_map[c]))
                if len(client_idx_map[richest]) > self.min_samples_per_client + needed:
                    donated = client_idx_map[richest][-needed:]
                    client_idx_map[richest] = client_idx_map[richest][:-needed]
                    client_idx_map[cid].extend(donated)

        for cid in client_ids:
            self.rng.shuffle(client_idx_map[cid])
            self.client_indices[cid] = client_idx_map[cid]

    def _partition_label_skew(self, client_ids: List[str]) -> None:
        """Each client is assigned a small subset of classes (label skew)."""
        client_idx_map = {cid: [] for cid in client_ids}
        classes_per_client = min(self.classes_per_client, self.num_classes)

        class_indices = {
            c: np.where(self.targets == c)[0] for c in range(self.num_classes)
        }
        for c in range(self.num_classes):
            self.rng.shuffle(class_indices[c])

        # Assign classes to each client in round-robin / randomized permutation
        for i, cid in enumerate(client_ids):
            assigned_classes = [
                (i * classes_per_client + k) % self.num_classes
                for k in range(classes_per_client)
            ]
            for c in assigned_classes:
                # Take a slice of class c indices
                c_pool = class_indices[c]
                quota = max(10, len(c_pool) // (self.num_clients * classes_per_client // self.num_classes + 1))
                selected = self.rng.choice(c_pool, size=min(quota, len(c_pool)), replace=False)
                client_idx_map[cid].extend(selected.tolist())

        for cid in client_ids:
            if len(client_idx_map[cid]) == 0:
                # Fallback to random samples
                fallback = self.rng.choice(len(self.targets), size=self.min_samples_per_client, replace=False)
                client_idx_map[cid] = fallback.tolist()
            self.rng.shuffle(client_idx_map[cid])
            self.client_indices[cid] = client_idx_map[cid]

    def get_client_subset(self, client_id: str) -> Subset:
        """Get the PyTorch Dataset Subset for a client."""
        return self.client_subsets[client_id]

    def get_client_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all clients."""
        return self.client_stats
