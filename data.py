import yfinance as yf
import pandas as pd

# Télécharger Tesla depuis Yahoo Finance
df = yf.download("TSLA", start="2005-01-01", end="2025-01-01", progress=False, auto_adjust=True)

df.to_excel("Tesla_2005_2025.xlsx")

print(df.head())
print(df.shape)


