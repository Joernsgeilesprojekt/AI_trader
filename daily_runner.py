"""Daily pipeline to fetch data, make prediction and log results."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

import pandas as pd

from src.data_loader import DataLoader
from src.news_processing import clean_text, embed_texts, sentiment_score
from src.feature_fusion import combine_daily
from src.model import PriceNewsModel
from src.predictor import load_model, predict
from src.indicators import rsi, macd
from src.backtester import backtest, StrategyConfig


def main() -> None:
    symbol = os.environ.get("SYMBOL", "AAPL")
    news_query = os.environ.get("NEWS_QUERY", symbol)
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise SystemExit("NEWS_API_KEY environment variable required")

    today = datetime.utcnow().date()
    start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    loader = DataLoader(news_api_key=api_key)
    prices = loader.load_prices(symbol, start, end)
    prices["RSI"] = rsi(prices["Close"]).fillna(0.0)
    macd_df = macd(prices["Close"])
    prices = prices.join(macd_df).fillna(0.0)

    news = loader.fetch_news(news_query, end, end, limit=20)
    texts = [clean_text(n.title + " " + n.description) for n in news]
    sentiments = [sentiment_score(t) for t in texts]
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

    actual = prices["Close"].iloc[-1]
    bt = backtest([actual, pred], [pred, pred], sentiments=[sentiments[-1] if sentiments else 0], strategy=StrategyConfig())
    profit = bt.profit

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "trades.csv")
    header = ["date", "prediction", "close", "profit", "sentiment", "headline"]
    headline = news[0].title if news else ""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        writer.writerow([end, f"{pred:.2f}", f"{actual:.2f}", f"{profit:.2f}", sentiments[0] if sentiments else 0, headline])
    print(f"Logged results to {log_path}")


if __name__ == "__main__":
    main()
