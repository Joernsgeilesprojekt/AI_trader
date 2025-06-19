"""Data loading utilities for price data and news articles."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as pdr



@dataclass
class NewsArticle:
    title: str
    description: str
    published_at: datetime


class DataLoader:
    """Utility class for fetching stock prices and news."""

    def __init__(self, news_api_key: Optional[str] = None) -> None:
        self.news_api_key = news_api_key

    def load_prices(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download historical prices from Yahoo Finance."""
        data = yf.download(symbol, start=start, end=end)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.dropna(inplace=True)
        return data

    def fetch_news(self, query: str, from_date: str, to_date: str, *, limit: int = 10) -> List[NewsArticle]:
        """Fetch news articles using the NewsAPI service."""
        if not self.news_api_key:
            raise ValueError("News API key not provided")
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={query}&from={from_date}&to={to_date}&pageSize={limit}&apiKey={self.news_api_key}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles: List[NewsArticle] = []
        for item in data.get("articles", []):
            articles.append(
                NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    published_at=datetime.fromisoformat(item["publishedAt"].replace("Z", "+00:00")),
                )
            )
        return articles

    def load_macro(self, series_id: str, start: str, end: str) -> pd.Series:
        """Load macroeconomic series from FRED."""
        return pdr.DataReader(series_id, "fred", start, end)
