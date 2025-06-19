# AI Trader

This repository contains modular components for stock price prediction using historical data combined with news sentiment.

Modules:
- `data_loader.py` – utilities to fetch historical price data from Yahoo Finance and query news articles.
- `news_processing.py` – text cleaning, sentiment analysis and embeddings.
- `model.py` – PyTorch models for price only or price + news prediction.
- `trainer.py` – training loops for the models.
- `predictor.py` – simple inference utility.

Example usage can be found within each module's docstrings.
