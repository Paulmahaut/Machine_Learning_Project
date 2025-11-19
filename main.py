import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from prophet import Prophet

# Dataset
df = yf.download("EURUSD=X", start="2005-01-01", end="2025-01-01")

# print(df.head())
# print(df.shape)

df_prophet = df.reset_index()[['Date','Close']] # Prepare data for Prophet
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
# mse = mean_squared_error(y_true, y_pred)
# rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("MAE:", mae)
# print("MSE:", mse)
# print("RMSE:", rmse)
print("MAPE:", mape, "%")

# Plot the graph
fig = model.plot(forecast)
plt.title("EUR/USD Forecast using Prophet (2005–2025)")
plt.xlabel("Date")
plt.ylabel("Exchange Rate (EUR/USD)")
plt.show()

plt.plot(test['ds'], y_true, label="Real")
plt.plot(pred['ds'], y_pred, label="Predicted")
plt.legend()
plt.show()
