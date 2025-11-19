"""
XGBoost Regression - Stock price prediction
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True):
    """
    Run XGBoost for price prediction
    
    Args:
        ticker: Stock symbol
        name: Display name
        prediction_days: Prediction horizon (default: 5 days)
        verbose: Print detailed output (default: True)
    
    Returns:
        dict with results (metrics, predictions, etc.)
    """
    
    # Download data
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    
    # Create features
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['Target'] = df['Close'].shift(-prediction_days)
    df = df.dropna()
    
    # Train/test split (80/20)
    features = ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility']
    X = df[features]
    y = df['Target']
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Training
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    if verbose:
        print(f"XGBoost Results: R²={r2:.4f} | RMSE=${rmse:.2f} | Train={len(X_train)}d | Test={len(X_test)}d")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(predictions, label='XGBoost', linewidth=2, alpha=0.7)
    plt.title(f'{name} - XGBoost Predictions ({prediction_days} days)\nR² = {r2:.4f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Test days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'{ticker}_xgboost.png'
    plt.savefig(filename, dpi=150)
    if verbose:
        print(f"Saved: {filename}")
    plt.show()
    
    return {
        'model': model,
        'predictions': predictions,
        'y_true': y_test.values,
        'metrics': {
            'RMSE': rmse,
            'R²': r2
        }
    }


if __name__ == "__main__":
    run_xgboost("TSLA", "Tesla")
