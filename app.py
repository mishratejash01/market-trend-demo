import streamlit as st
import pandas as pd
import plotly.express as px
from model import get_or_train, predict_future, train_lstm
from sentiment import get_sentiment_pipeline
import tempfile

st.set_page_config(page_title="AI Market Trend", layout="wide")
st.title("AI for Market Trend Analysis — LSTM + Sentiment")

st.write("Upload a CSV (date, sales) or use sample data below.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Using sample_data.csv")
    csv_path = "sample_data.csv"
else:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tfile.write(uploaded.getvalue())
    csv_path = tfile.name

st.sidebar.header("Settings")
seq_len = st.sidebar.slider("Sequence length", 7, 30, 14)
forecast_steps = st.sidebar.slider("Forecast days", 3, 30, 7)
epochs = st.sidebar.slider("Epochs", 5, 100, 20)
run_sentiment = st.sidebar.checkbox("Run sentiment (slow on first use)", False)
texts = st.sidebar.text_area("Review texts", "I love this product\nBad quality\nValue for money")

df = pd.read_csv(csv_path, parse_dates=[0])
st.dataframe(df.head())

fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Observed data")
st.plotly_chart(fig, use_container_width=True)

if st.button("Train model"):
    st.info("Training...")
    train_lstm(csv_path, seq_len, epochs)
    st.success("Model trained.")

model_path, scaler_path = get_or_train(csv_path, seq_len, epochs)

if st.button("Run forecast"):
    orig, forecast = predict_future(csv_path, model_path, scaler_path, seq_len, forecast_steps)
    fig2 = px.line()
    fig2.add_scatter(x=orig["date"], y=orig["value"], mode="lines+markers", name="Observed")
    fig2.add_scatter(x=forecast["date"], y=forecast["predicted"], mode="lines+markers", name="Forecast")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(forecast)

if run_sentiment:
    pipe = get_sentiment_pipeline()
    text_list = [t.strip() for t in texts.splitlines() if t.strip()]
    res = pipe(text_list)
    st.table(res)
