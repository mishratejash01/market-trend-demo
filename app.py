import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from model import get_or_train, predict_future
from sentiment import score_texts, get_sentiment_pipeline
from pytrends.request import TrendReq
import tempfile
import os

st.set_page_config(page_title="AI Market Trend Analysis", layout="wide")
st.title("AI for Market Trend Analysis")
st.markdown("An interactive dashboard for forecasting sales and analyzing market sentiment, aligned with the project guidelines.")

# --- File Uploader ---
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload your sales data (CSV)", type=["csv"])
if uploaded is None:
    st.sidebar.info("Using sample_data.csv for demonstration.")
    csv_path = "sample_data.csv"
else:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tfile.write(uploaded.getvalue())
    csv_path = tfile.name

df = pd.read_csv(csv_path, parse_dates=[0])
df.columns = ['date', 'sales'] # Standardize column names

# --- Main App Layout ---
tab1, tab2, tab3 = st.tabs(["Time Series Forecasting", "Sentiment Analysis", "Market Context (Google Trends)"])

# --- Tab 1: Time Series Forecasting ---
with tab1:
    st.header("Sales Forecasting with LSTM")
    st.write("This section uses an LSTM neural network to forecast future sales based on historical data.")

    # --- Sidebar Settings for Forecasting ---
    st.sidebar.header("1. Forecasting Settings")
    seq_len = st.sidebar.slider("Sequence length (days to look back)", 7, 60, 14, key="seq_len")
    forecast_steps = st.sidebar.slider("Forecast length (days to predict)", 3, 90, 30, key="forecast_steps")
    epochs = st.sidebar.slider("Training Epochs", 5, 200, 25, key="epochs")

    if st.button("Train Model and Generate Forecast", key="train_forecast"):
        with st.spinner("Training model and running forecast... This may take a moment."):
            model_path, scaler_path, history = get_or_train(csv_path, seq_len, epochs)
            orig, forecast, rmse, mae, eval_df = predict_future(csv_path, model_path, scaler_path, seq_len, forecast_steps)
            
            st.success("Model trained and forecast completed!")

            st.subheader("Forecast Results")
            col1, col2 = st.columns(2)
            col1.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}")
            col2.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
            
            fig_forecast = px.line()
            fig_forecast.add_scatter(x=orig["date"], y=orig["value"], mode="lines", name="Historical Data")
            fig_forecast.add_scatter(x=forecast["date"], y=forecast["predicted"], mode="lines+markers", name="Forecasted Data")
            fig_forecast.update_layout(title="Sales Forecast vs. Historical Data", xaxis_title="Date", yaxis_title="Sales")
            st.plotly_chart(fig_forecast, use_container_width=True)

            with st.expander("View Forecast Data Table"):
                st.dataframe(forecast)

            with st.expander("Model Performance on Historical Data"):
                fig_perf = px.line()
                fig_perf.add_scatter(x=eval_df["date"], y=eval_df["value"], mode='lines', name='Actual Values')
                fig_perf.add_scatter(x=eval_df["date"], y=eval_df['predicted_value'], mode='lines', name='Predicted Values')
                fig_perf.update_layout(title="Model Predictions vs. Actual Data (on training set)", xaxis_title="Date", yaxis_title="Sales")
                st.plotly_chart(fig_perf, use_container_width=True)

            if history:
                with st.expander("Training History"):
                    fig_hist, ax = plt.subplots()
                    ax.plot(history.history['loss'], label='Training Loss')
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                    ax.set_title('Model Training & Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (MSE)')
                    ax.legend()
                    st.pyplot(fig_hist)


# --- Tab 2: Sentiment Analysis ---
with tab2:
    st.header("Sentiment Analysis of Customer Reviews")
    st.write("Analyze customer reviews to gauge market sentiment. This uses a transformer-based NLP model.")
    
    st.sidebar.header("2. Sentiment Analysis Settings")
    texts_input = st.sidebar.text_area(
        "Enter review texts (one per line)", 
        "The product is amazing and works perfectly!\nI am very disappointed with the quality.\nIt's okay, not great but not bad either.\nBest purchase I've made all year.",
        height=150
    )
    
    if st.button("Analyze Sentiment", key="analyze_sentiment"):
        with st.spinner("Running sentiment analysis..."):
            text_list = [t.strip() for t in texts_input.splitlines() if t.strip()]
            if text_list:
                results_df = score_texts(text_list)
                st.subheader("Sentiment Results")
                st.dataframe(results_df)

                # Combine with sales data
                st.subheader("Sentiment Over Time (Example)")
                st.write("This chart plots the average sentiment score over the same period as your sales data. This is a simplified example; a real-world application would require timestamped reviews.")

                # Create a sentiment score (-1 for NEGATIVE, 1 for POSITIVE)
                results_df['score_numeric'] = results_df['label'].apply(lambda x: 1 if x == 'POSITIVE' else -1)
                avg_sentiment = results_df['score_numeric'].mean()
                
                sentiment_over_time = pd.DataFrame({
                    'date': df['date'],
                    'avg_sentiment': avg_sentiment
                })

                fig_sentiment_sales = px.line(df, x='date', y='sales', title="Sales vs. Average Sentiment")
                fig_sentiment_sales.add_scatter(x=sentiment_over_time['date'], y=sentiment_over_time['avg_sentiment'], name="Average Sentiment", yaxis="y2")
                fig_sentiment_sales.update_layout(
                    yaxis2=dict(
                        title="Average Sentiment Score",
                        overlaying="y",
                        side="right",
                        range=[-1.1, 1.1]
                    )
                )
                st.plotly_chart(fig_sentiment_sales, use_container_width=True)

            else:
                st.warning("Please enter some text to analyze.")

# --- Tab 3: Google Trends Analysis ---
with tab3:
    st.header("Market Context with Google Trends")
    st.write("Analyze public interest over time for specific keywords using Google Trends. Compare this with your sales data to find potential correlations.")

    st.sidebar.header("3. Google Trends Settings")
    keyword = st.sidebar.text_input("Enter a keyword to track", "organic snacks")

    if st.button("Fetch and Plot Google Trends", key="fetch_trends"):
        with st.spinner(f"Fetching Google Trends data for '{keyword}'..."):
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # Create a timeframe that matches the sales data
            start_date = df['date'].min().strftime('%Y-%m-%d')
            end_date = df['date'].max().strftime('%Y-%m-%d')
            timeframe = f"{start_date} {end_date}"

            try:
                pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
                trends_df = pytrends.interest_over_time()
                
                if not trends_df.empty:
                    st.subheader(f"Google Trends for '{keyword}'")
                    trends_df = trends_df.reset_index().rename(columns={'date': 'date', keyword: 'trend_score'})

                    fig_trends = px.line(trends_df, x='date', y='trend_score', title=f"Google Trends Interest Over Time for '{keyword}'")
                    st.plotly_chart(fig_trends, use_container_width=True)

                    # Combined plot
                    st.subheader("Sales vs. Google Trends Interest")
                    
                    fig_combined = px.line(df, x='date', y='sales', title=f"Sales vs. Google Trends for '{keyword}'")
                    fig_combined.add_scatter(x=trends_df['date'], y=trends_df['trend_score'], name="Google Trend Score", yaxis="y2")
                    fig_combined.update_layout(
                         yaxis=dict(title="Sales Volume"),
                         yaxis2=dict(
                            title="Google Trend Score (0-100)",
                            overlaying="y",
                            side="right"
                         )
                    )
                    st.plotly_chart(fig_combined, use_container_width=True)
                else:
                    st.warning("Could not retrieve Google Trends data for this keyword or timeframe. Please try another.")
            except Exception as e:
                st.error(f"An error occurred while fetching Google Trends data: {e}")

# Clean up temporary file
if 'tfile' in locals() and os.path.exists(tfile.name):
    os.remove(tfile.name)
