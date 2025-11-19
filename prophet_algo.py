"""
Prophet - Time series prediction
"""

import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def run_prophet(ticker="EURUSD=X", name="EUR/USD", test_years=3, verbose=True):
    """
    Run Prophet for price prediction
    
    Args:
        ticker: Stock/forex symbol
        name: Display name
        test_years: Years for test set (default: 3)
        verbose: Print detailed output (default: True)
    
    Returns:
        dict with results (metrics, predictions, etc.)
    """
    
    # Download data
    df = yf.download(ticker, start="2005-01-01", end="2025-01-01", progress=False)
    
    # Prepare for Prophet
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    
    # Train/test split
    test_days = test_years * 365
    train = df_prophet.iloc[:-test_days]
    test = df_prophet.iloc[-test_days:]
    
    # Training
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    model.fit(train)
    
    # Predictions
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    predictions = forecast[['ds', 'yhat']].iloc[-len(test):]
    
    # Metrics
    y_true = test['y'].values
    y_pred = predictions['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    if verbose:
        print(f"Prophet Results: R²={r2:.4f} | RMSE={rmse:.6f} | MAE={mae:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model.plot(forecast, ax=axes[0])
    axes[0].set_title(f'{name} - Prophet Forecast', fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    
    axes[1].plot(test['ds'], y_true, label='Actual', linewidth=2)
    axes[1].plot(predictions['ds'], y_pred, label='Prophet', linewidth=2, alpha=0.7)
    axes[1].set_title(f'Test Period Predictions\nR² = {r2:.4f}', fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{ticker.replace("=", "").replace("X", "")}_prophet.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"Saved: {filename}")
    plt.show()
    
    return {
        'model': model,
        'predictions': predictions,
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
    }


if __name__ == "__main__":
    run_prophet("EURUSD=X", "EUR/USD")
