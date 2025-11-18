import yfinance as yf
import pandas as pd

df = yf.download("EURUSD=X", start="2005-01-01", end="2025-01-01", progress=False)

df.to_excel("EURUSD_2005_2025.xlsx")
print("Fichier Excel créé : EURUSD_2005_2025.xlsx")
