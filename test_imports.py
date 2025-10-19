print("--- Starting import test ---")

print("Importing pandas...")
import pandas
print("Pandas imported.")

print("Importing plotly...")
import plotly
print("Plotly imported.")

print("Importing pytrends...")
from pytrends.request import TrendReq
print("Pytrends imported.")

print("Importing transformers (Hugging Face)...")
from transformers import pipeline
print("Transformers imported.")

print("Importing tensorflow (THIS IS THE SLOW ONE, PLEASE WAIT)...")
import tensorflow as tf
print(f"--- SUCCESS! TensorFlow version {tf.__version__} imported. ---")

print("\nAll libraries loaded. Your environment is correct.")
print("You can now run 'streamlit run app.py' again.")
