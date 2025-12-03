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
        print(f"\n{'='*80}")
        print(f"📊 {name} ({ticker}) - XGBoost v2 TESLA (24 optimized features)")
        print(f"{'='*80}")
        print(f"R²   = {r2:.4f}")
        print(f"RMSE = ${rmse:.2f}")
        print(f"MAE  = ${mae:.2f}")
        print(f"Train: {len(X_train)} days | Test: {len(X_test)} days")
        print(f"Prediction horizon: {prediction_days} day(s)")
        print(f"{'='*80}\n")
        
        # Feature importance
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("🎯 Top 10 Features by importance:")
        print(feature_imp_df.head(10).to_string(index=False))
        print()
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Predictions vs Real
    axes[0, 0].plot(y_test.values, label='Real Price', alpha=0.7, linewidth=2, color='blue')
    axes[0, 0].plot(predictions, label='Predictions', alpha=0.7, linestyle='--', linewidth=2, color='orange')
    axes[0, 0].set_title(f'{name} - Predictions vs Real Price', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Days (test set)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_test.values - predictions
    axes[0, 1].scatter(predictions, residuals, alpha=0.5, color='red')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residuals (Prediction Errors)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Predictions ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Feature Importance
    feature_imp_sorted = feature_imp_df.sort_values('Importance', ascending=True).tail(15)
    axes[1, 0].barh(feature_imp_sorted['Feature'], feature_imp_sorted['Importance'], color='skyblue')
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 4: Error Distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Error ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_xgboost_v2_tsla_optimized.png', dpi=150, bbox_inches='tight')
    
    if verbose:
        print(f"✅ Chart saved: {ticker}_xgboost_v2_tsla_optimized.png\n")
    
    plt.show()
    
    return {
        'model': model,
        'predictions': predictions,
        'y_test': y_test,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'feature_importance': feature_imp_df
    }


if __name__ == "__main__":
    # Test on Tesla
    print("\n🚀 XGBoost Simple v2 TESLA - Optimized for Tesla")
    print("Features: 24 (6 base + 18 TESLA-validated features)")
    
    results_tesla = run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True)
    
    print(f"\n{'='*80}")
    print("🎉 IMPROVEMENT vs Eval2:")
    print(f"{'='*80}")
    print(f"Eval2 (6 features):  R²=0.7874, RMSE=$26.80")
    print(f"v2 TSLA (24 features): R²={results_tesla['r2']:.4f}, RMSE=${results_tesla['rmse']:.2f}")
    print(f"Δ R²   = {results_tesla['r2'] - 0.7874:+.4f}  {'✅ BETTER!' if results_tesla['r2'] > 0.7874 else '❌ Worse'}")
    print(f"Δ RMSE = ${results_tesla['rmse'] - 26.80:+.2f}  {'✅ BETTER!' if results_tesla['rmse'] < 26.80 else '❌ Worse'}")
    print(f"{'='*80}\n")
