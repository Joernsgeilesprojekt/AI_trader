import os
import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.data_loader import DataLoader
from src.news_processing import clean_text, embed_texts
from src.feature_fusion import combine_daily
from src.indicators import rsi, macd
from src.model import PriceNewsModel
from src.predictor import load_model, predict
from src.backtester import backtest, StrategyConfig

import run_training
import run_prediction


class TraderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Trader GUI")

        tk.Label(self, text="News API Key").grid(row=0, column=0)
        tk.Label(self, text="Symbol").grid(row=1, column=0)
        tk.Label(self, text="News Query").grid(row=2, column=0)

        self.api_entry = tk.Entry(self, width=40)
        self.api_entry.grid(row=0, column=1)
        self.symbol_entry = tk.Entry(self, width=40)
        self.symbol_entry.insert(0, "AAPL")
        self.symbol_entry.grid(row=1, column=1)
        self.query_entry = tk.Entry(self, width=40)
        self.query_entry.grid(row=2, column=1)

        ttk.Button(self, text="Train", command=self.train_thread).grid(row=3, column=0, pady=5)
        ttk.Button(self, text="Predict", command=self.predict_thread).grid(row=3, column=1, pady=5)
        ttk.Button(self, text="Backtest", command=self.backtest_thread).grid(row=4, column=0, pady=5)

        self.pred_label = tk.Label(self, text="Predicted: N/A")
        self.pred_label.grid(row=4, column=1)

        fig = Figure(figsize=(5, 3))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)

    def get_inputs(self):
        api_key = self.api_entry.get().strip()
        symbol = self.symbol_entry.get().strip() or "AAPL"
        query = self.query_entry.get().strip() or symbol
        return api_key, symbol, query

    def train_thread(self):
        threading.Thread(target=self.train).start()

    def predict_thread(self):
        threading.Thread(target=self.predict).start()

    def backtest_thread(self):
        threading.Thread(target=self.run_backtest).start()

    def train(self):
        api, symbol, query = self.get_inputs()
        if not api:
            messagebox.showerror("Error", "API Key required")
            return
        start = (datetime.utcnow().date() - timedelta(days=180)).strftime("%Y-%m-%d")
        end = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            run_training.run_training(symbol, start, end, query, None, api)
            messagebox.showinfo("Training", "Training completed and model saved")
        except Exception as exc:
            messagebox.showerror("Training failed", str(exc))

    def predict(self):
        api, symbol, query = self.get_inputs()
        if not api:
            messagebox.showerror("Error", "API Key required")
            return
        try:
            pred = run_prediction.predict_latest(symbol, query, api)
            self.pred_label.config(text=f"Predicted next close: {pred:.2f}")
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))

    def run_backtest(self):
        api, symbol, query = self.get_inputs()
        if not api:
            messagebox.showerror("Error", "API Key required")
            return
        try:
            end = datetime.utcnow().date()
            start = end - timedelta(days=30)
            loader = DataLoader(news_api_key=api)
            prices = loader.load_prices(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            prices["RSI"] = rsi(prices["Close"]).fillna(0.0)
            macd_df = macd(prices["Close"])
            prices = prices.join(macd_df).fillna(0.0)
            news = loader.fetch_news(query, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), limit=50)
            texts = [clean_text(n.title + " " + n.description) for n in news]
            if texts:
                embeddings = embed_texts(texts)
                news_df = pd.DataFrame(embeddings, index=[n.published_at for n in news])
            else:
                news_df = pd.DataFrame([], columns=[0])
            data = combine_daily(prices, news_df)
            X = data.iloc[:-1].values
            price_dim = prices.shape[1]
            model = PriceNewsModel(price_dim=price_dim, news_dim=data.shape[1] - price_dim)
            load_model("models/model.pt", model)
            preds = predict(model, X)
            result = backtest(prices["Close"].values[:len(preds)], preds, strategy=StrategyConfig())
            msg = f"Sharpe: {result.sharpe:.2f}\nProfit: {result.profit:.2f}\nMax DD: {result.max_drawdown:.2f}"\
                f"\nWin/Loss: {result.win_loss_ratio:.2f}"
            messagebox.showinfo("Backtest", msg)
            self.ax.clear()
            self.ax.plot(prices.index[:-1], prices["Close"].values[:-1], label="Price")
            self.ax.plot(prices.index[:-1], pd.Series(preds, index=prices.index[:-1]), label="Pred")
            self.ax.legend()
            self.canvas.draw()
        except FileNotFoundError:
            messagebox.showwarning("Backtest", "Model not found. Train first.")
        except Exception as exc:
            messagebox.showerror("Backtest failed", str(exc))


if __name__ == "__main__":
    app = TraderGUI()
    app.mainloop()
