import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


df = yf.download("EURUSD=X", start="2005-01-01", end="2025-01-01")

# print(df.head())
# print(df.shape)

df_prophet = df.reset_index()[['Date','Close']]
df_prophet.columns = ['ds','y']

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)


fig = model.plot(forecast)
plt.title("EUR/USD Forecast using Prophet (2005–2025)")
plt.xlabel("Date")
plt.ylabel("Exchange Rate (EUR/USD)")
plt.show()