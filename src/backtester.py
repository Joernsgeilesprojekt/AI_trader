"""Evaluate predictions vs actual prices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
from typing import Iterable, List

import numpy as np


@dataclass
class BacktestResult:
    sharpe: float
    profit: float
    max_drawdown: float
    returns: List[float]
    win_loss_ratio: float


@dataclass
class StrategyConfig:
    """Configuration for trading strategy."""

    threshold: float = 0.0
    stop_loss: Optional[float] = None
    news_only: bool = False


def backtest(
    prices: Iterable[float],
    predictions: Iterable[float],
    *,
    sentiments: Optional[Iterable[float]] = None,
    strategy: StrategyConfig | None = None,
) -> BacktestResult:
    """Backtest predictions using a configurable strategy."""



def backtest(prices: Iterable[float], predictions: Iterable[float]) -> BacktestResult:
    """Simple backtesting using daily close prices and predicted next-day closes."""
    prices = np.asarray(prices)
    preds = np.asarray(predictions)
    if len(prices) != len(preds):
        raise ValueError("prices and predictions must have same length")

    if strategy is None:
        strategy = StrategyConfig()

    daily_returns = []

    daily_returns = []
    position = 0.0
    cash = 0.0
    for i in range(len(prices) - 1):
        today_price = prices[i]
        next_price = prices[i + 1]

        if strategy.news_only and sentiments is not None:
            sentiment_ok = sentiments[i] > 0
        else:
            sentiment_ok = True

        long_signal = preds[i] - today_price > strategy.threshold
        if long_signal and sentiment_ok:
            daily_return = next_price - today_price
            if strategy.stop_loss is not None and daily_return < -strategy.stop_loss:
                daily_return = -strategy.stop_loss
        else:
            daily_return = 0.0


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
    win_trades = (returns > 0).sum()
    loss_trades = (returns < 0).sum()
    win_loss_ratio = win_trades / loss_trades if loss_trades else float("inf")

    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0
    return BacktestResult(
        sharpe=sharpe,
        profit=cash,
        max_drawdown=max_drawdown,
        returns=daily_returns,
        win_loss_ratio=win_loss_ratio,
    )


__all__ = ["backtest", "BacktestResult", "StrategyConfig"]

    return BacktestResult(sharpe=sharpe, profit=cash, max_drawdown=max_drawdown, returns=daily_returns)


__all__ = ["backtest", "BacktestResult"]
