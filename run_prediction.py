"""Fetch latest news and make a prediction."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from src.data_loader import DataLoader
from src.news_processing import clean_text, embed_texts
from src.feature_fusion import combine_daily
from src.model import PriceNewsModel
from src.predictor import load_model, predict
from src.indicators import rsi, macd


def predict_latest(symbol: str, news_query: str, api_key: str) -> float:



def main() -> None:
    symbol = os.environ.get("SYMBOL", "AAPL")
    news_query = os.environ.get("NEWS_QUERY", symbol)
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise SystemExit("NEWS_API_KEY environment variable required")

    end = datetime.utcnow().date()
    start = (end - timedelta(days=30)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    loader = DataLoader(news_api_key=api_key)
    prices = loader.load_prices(symbol, start, end_str)
    prices["RSI"] = rsi(prices["Close"]).fillna(0.0)
    macd_df = macd(prices["Close"])
    prices = prices.join(macd_df).fillna(0.0)
    news = loader.fetch_news(news_query, end_str, end_str, limit=10)

    texts = [clean_text(n.title + " " + n.description) for n in news]
    if texts:
        embeddings = embed_texts(texts)
        news_df = pd.DataFrame(embeddings, index=[n.published_at for n in news])
    else:
        news_df = pd.DataFrame([], columns=[0])

    data = combine_daily(prices, news_df)
    X = data.values[-1:]

    price_dim = prices.shape[1]
    model = PriceNewsModel(price_dim=price_dim, news_dim=data.shape[1] - price_dim)
    load_model("models/model.pt", model)
    pred = predict(model, X)[0]
    return pred


def main() -> None:
    symbol = os.environ.get("SYMBOL", "AAPL")
    news_query = os.environ.get("NEWS_QUERY", symbol)
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise SystemExit("NEWS_API_KEY environment variable required")

    pred = predict_latest(symbol, news_query, api_key)

    model = PriceNewsModel(price_dim=5, news_dim=data.shape[1] - 5)
    load_model("models/model.pt", model)
    pred = predict(model, X)[0]
    print(f"Predicted next close for {symbol}: {pred:.2f}")


if __name__ == "__main__":
    main()
