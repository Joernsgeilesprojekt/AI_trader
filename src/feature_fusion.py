"""Utilities for combining price data with news embeddings."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def combine_daily(prices: pd.DataFrame, news_embeddings: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with price features joined with daily mean news embeddings."""
    price_df = prices.copy()
    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise ValueError("prices index must be DatetimeIndex")

    news_df = news_embeddings.copy()
    if not isinstance(news_df.index, pd.DatetimeIndex):
        raise ValueError("news index must be DatetimeIndex")

    news_daily = news_df.groupby(news_df.index.date).mean()
    news_daily.index = pd.to_datetime(news_daily.index)

    merged = price_df.join(news_daily, how="left").fillna(0.0)
    return merged


__all__ = ["combine_daily"]
