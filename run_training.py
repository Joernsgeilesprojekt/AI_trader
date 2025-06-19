"""Run end-to-end training using prices and news."""
from __future__ import annotations

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


def run_training(
    symbol: str,
    start: str,
    end: str,
    news_query: str,
    fred_series: str | None,
    api_key: str,
) -> None:

    loader = DataLoader(news_api_key=api_key)
    prices = loader.load_prices(symbol, start, end)
    prices["RSI"] = rsi(prices["Close"]).fillna(0.0)
    macd_df = macd(prices["Close"])
    prices = prices.join(macd_df).fillna(0.0)
    if fred_series:
        try:
            macro = loader.load_macro(fred_series, start, end)
            prices = prices.join(macro.rename(fred_series)).fillna(method="ffill")
        except Exception as exc:
            print(f"Failed to fetch macro data: {exc}")

def main() -> None:
    symbol = os.environ.get("SYMBOL", "AAPL")
    start = os.environ.get("START", "2024-01-01")
    end = os.environ.get("END", datetime.utcnow().strftime("%Y-%m-%d"))
    news_query = os.environ.get("NEWS_QUERY", symbol)
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise SystemExit("NEWS_API_KEY environment variable required")

    loader = DataLoader(news_api_key=api_key)
    prices = loader.load_prices(symbol, start, end)
    news = loader.fetch_news(news_query, start, end, limit=100)

    texts = [clean_text(n.title + " " + n.description) for n in news]
    embeddings = embed_texts(texts)
    news_df = pd.DataFrame(embeddings, index=[n.published_at for n in news])

    data = combine_daily(prices, news_df)
    X = data.iloc[:-1].values
    y = data["Close"].shift(-1).dropna().values

    price_dim = prices.shape[1]
    model = PriceNewsModel(price_dim=price_dim, news_dim=embeddings.shape[1])
    model = PriceNewsModel(price_dim=5, news_dim=embeddings.shape[1])
    train_model(model, X, y, TrainConfig(epochs=3))

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")
    print("Model saved to models/model.pt")


def main() -> None:
    symbol = os.environ.get("SYMBOL", "AAPL")
    start = os.environ.get("START", "2024-01-01")
    end = os.environ.get("END", datetime.utcnow().strftime("%Y-%m-%d"))
    news_query = os.environ.get("NEWS_QUERY", symbol)
    fred_series = os.environ.get("FRED_SERIES")
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise SystemExit("NEWS_API_KEY environment variable required")

    run_training(symbol, start, end, news_query, fred_series, api_key)


if __name__ == "__main__":
    main()
