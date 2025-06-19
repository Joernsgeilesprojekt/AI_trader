"""Simple grid search for training hyperparameters."""
from __future__ import annotations

import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.data_loader import DataLoader
from src.news_processing import clean_text, embed_texts
from src.feature_fusion import combine_daily
from src.model import PriceNewsModel
from src.trainer import TrainConfig, train_model
from src.indicators import rsi, macd


def main() -> None:
    symbol = os.environ.get("SYMBOL", "AAPL")
    start = os.environ.get("START", "2024-01-01")
    end = os.environ.get("END", datetime.utcnow().strftime("%Y-%m-%d"))
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise SystemExit("NEWS_API_KEY environment variable required")

    loader = DataLoader(news_api_key=api_key)
    prices = loader.load_prices(symbol, start, end)
    prices["RSI"] = rsi(prices["Close"]).fillna(0.0)
    macd_df = macd(prices["Close"])
    prices = prices.join(macd_df).fillna(0.0)

    news = loader.fetch_news(symbol, start, end, limit=100)
    texts = [clean_text(n.title + " " + n.description) for n in news]
    embeddings = embed_texts(texts)
    news_df = pd.DataFrame(embeddings, index=[n.published_at for n in news])

    data = combine_daily(prices, news_df)
    X = data.iloc[:-1].values
    y = data["Close"].shift(-1).dropna().values

    price_dim = prices.shape[1]
    news_dim = embeddings.shape[1]

    hidden_dims = [32, 64]
    lrs = [1e-3, 5e-4]
    window_sizes = [30, 60]

    results = []
    for hd, lr, win in itertools.product(hidden_dims, lrs, window_sizes):
        model = PriceNewsModel(price_dim=price_dim, news_dim=news_dim, hidden_dim=hd)
        cfg = TrainConfig(lr=lr, epochs=2)
        train_model(model, X[-win:], y[-win:], cfg)
        preds = model(torch.tensor(X[-win:], dtype=torch.float32)).detach().numpy().squeeze()
        mse = ((preds - y[-win:]) ** 2).mean()
        results.append({"hidden_dim": hd, "lr": lr, "window": win, "mse": mse})
        print(f"hd={hd}, lr={lr}, window={win}, mse={mse:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("hyperparam_results.csv", index=False)
    print("Saved results to hyperparam_results.csv")


if __name__ == "__main__":
    main()
