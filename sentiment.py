from transformers import pipeline
import pandas as pd

def get_sentiment_pipeline(device=-1):
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

def score_texts(texts, sentiment_pipe=None):
    if sentiment_pipe is None:
        sentiment_pipe = get_sentiment_pipeline()
    results = sentiment_pipe(texts, batch_size=16)
    return pd.DataFrame(results)
