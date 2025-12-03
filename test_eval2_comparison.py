"""
Test comparison - xgboost_simple (Eval2 branch) on TotalEnergies
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def run_xgboost_eval2(ticker="TTE.PA", name="TotalEnergies", prediction_days=5):
    # Download data
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Create features
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['RoC_5'] = df['Close'].pct_change(periods=5) * 100
    df['RoC_10'] = df['Close'].pct_change(periods=10) * 100
    
    df['Volume_MA'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Target
    df['Target'] = df['Close'].shift(-prediction_days)
    df = df.dropna()
    
    # Features (11 features comme dans Eval2)
    features = ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility',
                'RSI', 'RoC_5', 'RoC_10', 'Volume_MA', 'Volume_Ratio']
    
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
    
    print(f"\n{'='*80}")
    print(f"📊 {name} ({ticker}) - XGBoost EVAL2 (11 features)")
    print(f"{'='*80}")
    print(f"R²   = {r2:.4f}")
    print(f"RMSE = ${rmse:.2f}")
    print(f"MAE  = ${mae:.2f}")
    print(f"Train: {len(X_train)} days | Test: {len(X_test)} days")
    print(f"Features: {features}")
    print(f"{'='*80}\n")
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}

if __name__ == "__main__":
    run_xgboost_eval2("TTE.PA", "TotalEnergies", prediction_days=5)
