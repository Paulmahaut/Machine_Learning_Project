"""
XGBoost Simple v2 TESLA - Optimized version with TESLA-specific features
Based on test_all_features_tsla.py analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True):

    # Download data
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # ========== BASE FEATURES (from xgboost_simple.py) ==========
    # Moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    
    # Price lags (historical prices)
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # ========== TESLA-OPTIMIZED FEATURES (tested and validated) ==========
    # ✅ Bollinger Bands (best improvement: +0.0425 R²)
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # ✅ Price Changes (improvement: +0.0420 R²)
    df['Price_change_1d'] = df['Close'].diff(1)
    df['Price_change_3d'] = df['Close'].diff(3)
    df['Price_pct_1d'] = df['Close'].pct_change(1)
    df['Price_pct_3d'] = df['Close'].pct_change(3)
    
    # ✅ Rate of Change (improvement: +0.0333 R²)
    df['RoC_5'] = df['Close'].pct_change(5)
    df['RoC_10'] = df['Close'].pct_change(10)
    
    # ✅ Momentum (improvement: +0.0313 R²)
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # ✅ MACD (improvement: +0.0201 R²)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # ✅ ATR (improvement: +0.0179 R²)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # ✅ Volume (improvement: +0.0159 R²)
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    # Target variable (price to predict)
    df['Target'] = df['Close'].shift(-prediction_days)
    df = df.dropna()
    
    # Train/test split (80/20)
    features = [
        # Base features (6)
        'MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility',
        # Bollinger Bands (2)
        'BB_width', 'BB_position',
        # Price Changes (4)
        'Price_change_1d', 'Price_change_3d', 'Price_pct_1d', 'Price_pct_3d',
        # RoC (2)
        'RoC_5', 'RoC_10',
        # Momentum (2)
        'Momentum_5', 'Momentum_10',
        # MACD (3)
        'MACD', 'MACD_signal', 'MACD_hist',
        # ATR (1)
        'ATR',
        # Volume (2)
        'Volume_MA_5', 'Volume_Ratio',
        # EMA (2)
        'EMA_12', 'EMA_26'
    ]
    # Total: 24 TESLA-optimized features
     
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
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    if verbose:
        print(f"XGBoost Results: R²={r2:.4f} | RMSE=${rmse:.2f} | Train={len(X_train)}d | Test={len(X_test)}d")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Real Price', alpha=0.7, linewidth=2)
    plt.plot(predictions, label='Predictions', alpha=0.7, linestyle='--', linewidth=2)
    plt.title(f'{name} - Stock Price Prediction (R²={r2:.4f}) with 24 features', fontsize=14, fontweight='bold')
    plt.xlabel('Days (test set)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{ticker}_xgboost_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'model': model,
        'predictions': predictions,
        'y_test': y_test,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


if __name__ == "__main__":
    # Test on Tesla
    results_tesla = run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True)

