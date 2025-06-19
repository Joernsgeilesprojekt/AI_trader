"""Utilities for processing news text and generating embeddings."""
from __future__ import annotations

import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

stop_words = set(stopwords.words('english'))

_sia = SentimentIntensityAnalyzer()
_model = SentenceTransformer('all-MiniLM-L6-v2')


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)


def sentiment_score(text: str) -> float:
    """Return compound VADER sentiment score."""
    return _sia.polarity_scores(text)['compound']


def embed_texts(texts: Iterable[str]):
    """Return sentence embeddings for given texts."""
    return _model.encode(list(texts))


__all__ = [
    "clean_text",
    "sentiment_score",
    "embed_texts",
]
