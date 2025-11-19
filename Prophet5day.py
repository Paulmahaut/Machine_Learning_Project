import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from prophet import Prophet

# Dataset
df = yf.download("TTE.PA", start="2005-01-01", end="2025-01-01", progress=False)

df["Close_5d"] = df["Close"].rolling(window=5).mean()

df = df.dropna()

df_prophet = df.reset_index()[['Date','Close_5d']] # Prepare data for Prophet
df_prophet.columns = ['ds','y'] # ds: date, y: value (seulement 2 colonnes)

# training and testing split
train = df_prophet.iloc[:-1095]  
test = df_prophet.iloc[-1095:]   # on prend 3 ans pour avoir ~20% de test

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(len(test))
forecast = model.predict(future)

pred = forecast[['ds','yhat']].iloc[-len(test):]

# Accuracy metrics
y_true = test['y'].values
y_pred = pred['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape, "%")

# Plot the graph
fig = model.plot(forecast)
plt.title("TTE.PA - Prophet Forecast on 5-day Rolling Mean")
plt.xlabel("Date")
plt.ylabel("5-day Average Price (€)")
plt.show()

# Comparison plot (real vs predicted)
plt.figure(figsize=(12,5))
plt.plot(test['ds'], y_true, label="Real (5d avg)")
plt.plot(pred['ds'], y_pred, label="Predicted (5d avg)")
plt.legend()
plt.title("Comparison: Real vs Predicted (5d Rolling Mean)")
plt.show()
