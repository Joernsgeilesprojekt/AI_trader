"""Training utilities for price prediction models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32


def train_model(model: nn.Module, X, y, config: TrainConfig) -> nn.Module:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    optimiser = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(config.epochs):
        for xb, yb in loader:
            optimiser.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds.squeeze(), yb)
            loss.backward()
            optimiser.step()
    return model
