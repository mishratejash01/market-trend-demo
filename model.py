import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os, joblib

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_series(csv_path, date_col="date", value_col="sales"):
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col)[[date_col, value_col]].dropna()
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df = df.reset_index(drop=True)
    return df

def create_sequences(values, seq_len=14):
    x, y = [], []
    for i in range(len(values) - seq_len):
        x.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(x), np.array(y)

def train_lstm(csv_path, seq_len=14, epochs=30, batch_size=8, model_name="lstm_model"):
    df = load_series(csv_path)
    values = df["value"].values.reshape(-1, 1).astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = create_sequences(scaled, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es])

    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.save")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path, df

def predict_future(csv_path, model_path, scaler_path, seq_len=14, n_steps=7):
    df = load_series(csv_path)
    values = df["value"].values.reshape(-1, 1).astype("float32")
    scaler = joblib.load(scaler_path)
    from tensorflow.keras.models import load_model
    model = load_model(model_path, compile=False)


    scaled = scaler.transform(values)
    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    preds = []
    current_seq = last_seq.copy()
    for _ in range(n_steps):
        p = model.predict(current_seq, verbose=0)[0][0]
        preds.append(p)
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = p

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    last_date = pd.to_datetime(df["date"].iloc[-1])
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_steps)]
    forecast_df = pd.DataFrame({"date": future_dates, "predicted": preds_inv})
    return df, forecast_df

def get_or_train(csv_path, seq_len=14, epochs=30):
    model_name = "lstm_market"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.save")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return model_path, scaler_path
    else:
        return train_lstm(csv_path, seq_len, epochs, 8, model_name)[:2]
