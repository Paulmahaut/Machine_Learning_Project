"""
XGBoost Simple v2 - Optimized version with relevant features
Based on xgboost_simple.py + features validated by test_all_features.py
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
    
    # ========== NEW RELEVANT FEATURES ==========
    # ✅ Rolling Statistics (best improvement: +0.0498 R²)
    df['Rolling_std_20'] = df['Close'].rolling(20).std()
    df['Rolling_min_20'] = df['Close'].rolling(20).min()
    df['Rolling_max_20'] = df['Close'].rolling(20).max()
    df['Rolling_skew_20'] = df['Close'].rolling(20).skew()
    df['Rolling_kurt_20'] = df['Close'].rolling(20).kurt()
    
    # ✅ RSI - Relative Strength Index (improvement: +0.0453 R²)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ✅ Additional Volatility (improvement: +0.0170 R²)
    df['Volatility_5'] = df['Close'].rolling(5).std()
    df['Volatility_20_alt'] = df['Close'].rolling(20).std()
    
    # Target variable (price to predict)
    df['Target'] = df['Close'].shift(-prediction_days)
    df = df.dropna()
    
    # Train/test split (80/20)
    features = [
        # Base features (6)
        'MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility',
        # Rolling Stats (5)
        'Rolling_std_20', 'Rolling_min_20', 'Rolling_max_20', 'Rolling_skew_20', 'Rolling_kurt_20',
        # RSI (1)
        'RSI',
        # Additional Volatility (2)
        'Volatility_5', 'Volatility_20_alt'
    ]
    # Total: 14 optimal features
     
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
        print(f"📊 {name} ({ticker}) - XGBoost v2 (14 optimized features)")
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
    feature_imp_sorted = feature_imp_df.sort_values('Importance', ascending=True)
    axes[1, 0].barh(feature_imp_sorted['Feature'], feature_imp_sorted['Importance'], color='skyblue')
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title('Feature Importance (14 features)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 4: Error Distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Error ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_xgboost_v2_results.png', dpi=150, bbox_inches='tight')
    
    if verbose:
        print(f"✅ Chart saved: {ticker}_xgboost_v2_results.png\n")
    
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
    # Test on TotalEnergies
    print("\n🚀 XGBoost Simple v2 - Optimized Version")
    print("Features: 14 (6 base + 8 new relevant features)")
    
    results_tte = run_xgboost(ticker="TTE.PA", name="TotalEnergies", prediction_days=5, verbose=True)
    
    # Optional: test on other assets
    # results_tesla = run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True)
    # results_apple = run_xgboost(ticker="AAPL", name="Apple", prediction_days=5, verbose=True)
