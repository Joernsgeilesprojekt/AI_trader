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

    model = PriceNewsModel(price_dim=5, news_dim=embeddings.shape[1])
    train_model(model, X, y, TrainConfig(epochs=3))

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")
    print("Model saved to models/model.pt")


if __name__ == "__main__":
    main()
