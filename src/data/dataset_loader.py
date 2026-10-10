"""Dataset loader for Privara federated learning.

Supports MNIST, Fashion-MNIST, and a robust offline Synthetic dataset fallback.
Ensures zero data leakage between client training subsets and global test evaluation.
"""

import os
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from utils.logging import logger, log_event


class SyntheticDataset(Dataset):
    """Synthetic 10-class 28x28 image dataset for offline or rapid testing.
    
    Generates class-dependent geometric patterns with Gaussian noise.
    """
    def __init__(self, num_samples: int = 5000, num_classes: int = 10, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.num_classes = num_classes
        
        # Generate balanced class labels
        labels = np.repeat(np.arange(num_classes), num_samples // num_classes)
        if len(labels) < num_samples:
            extra = rng.choice(num_classes, size=num_samples - len(labels))
            labels = np.concatenate([labels, extra])
        rng.shuffle(labels)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Generate 1x28x28 class-dependent synthetic images
        images = []
        for label in labels:
            base = np.zeros((28, 28), dtype=np.float32)
            # Center shape based on class
            r_start = (label % 4) * 5 + 4
            c_start = (label // 4) * 7 + 4
            base[r_start:r_start + 8, c_start:c_start + 8] = 0.8
            # Add Gaussian noise
            noise = rng.normal(0.0, 0.2, (28, 28)).astype(np.float32)
            img = np.clip(base + noise, 0.0, 1.0)
            images.append(img[np.newaxis, ...])
        
        self.images = torch.tensor(np.array(images), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def load_dataset(
    dataset_name: str = "MNIST",
    data_dir: str = "./data",
    max_train_samples: Optional[int] = 10000,
    max_test_samples: Optional[int] = 2000,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Load benchmark dataset (MNIST / FashionMNIST) or synthetic fallback.
    
    Returns:
        (train_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = None
    test_dataset = None
    
    if dataset_name.upper() in ["MNIST", "FASHIONMNIST"]:
        dataset_cls = (
            torchvision.datasets.MNIST
            if dataset_name.upper() == "MNIST"
            else torchvision.datasets.FashionMNIST
        )
        try:
            logger.info(f"Attempting to load {dataset_name} from {data_dir}...")
            train_raw = dataset_cls(root=data_dir, train=True, download=True, transform=transform)
            test_raw = dataset_cls(root=data_dir, train=False, download=True, transform=transform)
            
            # Subsample if requested for fast CPU training
            if max_train_samples and max_train_samples < len(train_raw):
                rng = np.random.RandomState(seed)
                train_idx = rng.choice(len(train_raw), max_train_samples, replace=False)
                # Extract tensors for deterministic fast indexing
                train_images = torch.stack([train_raw[i][0] for i in train_idx])
                train_labels = torch.tensor([train_raw[i][1] for i in train_idx], dtype=torch.long)
                train_dataset = TensorDataset(train_images, train_labels)
            else:
                train_images = torch.stack([train_raw[i][0] for i in range(len(train_raw))])
                train_labels = torch.tensor([train_raw[i][1] for i in range(len(train_raw))], dtype=torch.long)
                train_dataset = TensorDataset(train_images, train_labels)

            if max_test_samples and max_test_samples < len(test_raw):
                rng = np.random.RandomState(seed + 1)
                test_idx = rng.choice(len(test_raw), max_test_samples, replace=False)
                test_images = torch.stack([test_raw[i][0] for i in test_idx])
                test_labels = torch.tensor([test_raw[i][1] for i in test_idx], dtype=torch.long)
                test_dataset = TensorDataset(test_images, test_labels)
            else:
                test_images = torch.stack([test_raw[i][0] for i in range(len(test_raw))])
                test_labels = torch.tensor([test_raw[i][1] for i in range(len(test_raw))], dtype=torch.long)
                test_dataset = TensorDataset(test_images, test_labels)

            log_event("DATASET_LOAD", f"Loaded {dataset_name}: {len(train_dataset)} train, {len(test_dataset)} test samples.")
            return train_dataset, test_dataset

        except Exception as e:
            logger.warning(f"Could not download {dataset_name} ({e}). Falling back to SyntheticDataset.")
            log_event("DATASET_FALLBACK", f"Fallback to SyntheticDataset due to download issue: {str(e)[:100]}", level="warning")
    
    # Synthetic dataset fallback
    train_n = max_train_samples or 6000
    test_n = max_test_samples or 1500
    train_dataset = SyntheticDataset(num_samples=train_n, num_classes=10, seed=seed)
    test_dataset = SyntheticDataset(num_samples=test_n, num_classes=10, seed=seed + 999)
    log_event("DATASET_LOAD", f"Loaded SyntheticDataset: {train_n} train, {test_n} test samples.")
    return train_dataset, test_dataset
