import yfinance as yf
import pandas as pd

# Télécharger TotalEnergies depuis Yahoo Finance
df = yf.download("TTE.PA", start="2005-01-01", end="2025-01-01", progress=False)

df.to_excel("TotalEnergies_2005_2025.xlsx")

print(df.head())
print(df.shape)


