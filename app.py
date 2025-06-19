"""Simple Streamlit dashboard for AI Trader."""
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src.data_loader import DataLoader
from src.news_processing import clean_text, sentiment_score
from src.predictor import load_model, predict
from src.feature_fusion import combine_daily
from src.model import PriceNewsModel
from src.indicators import rsi, macd
from src.backtester import backtest, StrategyConfig

st.title("AI Trader Dashboard")
api_key = st.sidebar.text_input("News API Key", value=os.environ.get("NEWS_API_KEY", ""))
symbol = st.sidebar.text_input("Symbol", value="AAPL")
query = st.sidebar.text_input("News Query", value=symbol)

if api_key:
    loader = DataLoader(news_api_key=api_key)
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)
    prices = loader.load_prices(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    prices["RSI"] = rsi(prices["Close"]).fillna(0.0)
    macd_df = macd(prices["Close"])
    prices = prices.join(macd_df).fillna(0.0)

    news = loader.fetch_news(query, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), limit=20)

    st.subheader("Price Chart")
    st.line_chart(prices["Close"])

    st.subheader("News and Sentiment")
    news_data = []
    for n in news:
        cleaned = clean_text(n.title + " " + n.description)
        sent = sentiment_score(cleaned)
        news_data.append({"title": n.title, "date": n.published_at, "sentiment": sent})
    news_df = pd.DataFrame(news_data)
    st.table(news_df)

    st.subheader("Prediction")
    texts = [clean_text(n.title + " " + n.description) for n in news]
    if texts:
        import numpy as np
        from src.news_processing import embed_texts

        embeddings = embed_texts(texts)
        emb_df = pd.DataFrame(embeddings, index=[n.published_at for n in news])
    else:
        emb_df = pd.DataFrame([], columns=[0])
    data = combine_daily(prices, emb_df)
    X = data.values[-1:]

    price_dim = prices.shape[1]
    model = PriceNewsModel(price_dim=price_dim, news_dim=data.shape[1] - price_dim)

    model = PriceNewsModel(price_dim=5, news_dim=data.shape[1] - 5)
    try:
        load_model("models/model.pt", model)
        pred = predict(model, X)[0]
        st.write(f"Predicted next close: {pred:.2f}")

        preds_hist = predict(model, data.values[:-1])
        bt = backtest(prices["Close"].values[: len(preds_hist)], preds_hist, strategy=StrategyConfig())
        st.subheader("Backtest Metrics")
        st.write(f"Sharpe Ratio: {bt.sharpe:.2f}")
        st.write(f"Cumulative Profit: {bt.profit:.2f}")
        st.write(f"Max Drawdown: {bt.max_drawdown:.2f}")
        st.write(f"Win/Loss Ratio: {bt.win_loss_ratio:.2f}")
        import numpy as np
        cumulative = np.cumsum(bt.returns)
        st.line_chart(cumulative)
    except FileNotFoundError:
        st.warning("Trained model not found. Run run_training.py first.")
else:
    st.info("Enter News API key to begin")
