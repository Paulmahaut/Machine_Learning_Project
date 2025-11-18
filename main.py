import yfinance as yf

df = yf.download("EURUSD=X", start="2005-01-01", end="2025-01-01")
print(df.head())
print(df.shape)
