"""Evaluate predictions vs actual prices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class BacktestResult:
    sharpe: float
    profit: float
    max_drawdown: float
    returns: List[float]


def backtest(prices: Iterable[float], predictions: Iterable[float]) -> BacktestResult:
    """Simple backtesting using daily close prices and predicted next-day closes."""
    prices = np.asarray(prices)
    preds = np.asarray(predictions)
    if len(prices) != len(preds):
        raise ValueError("prices and predictions must have same length")

    daily_returns = []
    position = 0.0
    cash = 0.0
    for i in range(len(prices) - 1):
        today_price = prices[i]
        next_price = prices[i + 1]
        if preds[i] > today_price:
            # go long for one day
            daily_return = next_price - today_price
        else:
            daily_return = 0.0
        cash += daily_return
        daily_returns.append(daily_return)
    returns = np.array(daily_returns)
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0
    return BacktestResult(sharpe=sharpe, profit=cash, max_drawdown=max_drawdown, returns=daily_returns)


__all__ = ["backtest", "BacktestResult"]
