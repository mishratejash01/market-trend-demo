import pandas as pd

def load_csv_preview(path, n=5):
    df = pd.read_csv(path, parse_dates=[0])
    return df.head(n)
