"""PyTorch models for price and news prediction."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PriceLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class PriceNewsModel(nn.Module):
    """Combine price sequences with news embeddings."""

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, num_layers=1, batch_first=True)
        self.news_fc = nn.Linear(news_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, price_seq, news_vec):
        p, _ = self.price_lstm(price_seq)
        p = p[:, -1, :]
        n = self.news_fc(news_vec)
        x = torch.cat([p, n], dim=1)
        return self.out(x)
