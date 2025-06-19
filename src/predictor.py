"""Inference helper for trained models."""
from __future__ import annotations

import torch
from torch import nn


def load_model(path: str, model: nn.Module) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: nn.Module, X):
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = model(X_t)
        return preds.squeeze().numpy()
