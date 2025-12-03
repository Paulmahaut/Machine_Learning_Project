"""
Check for overfitting - Compare train vs test performance
XGBoost v1 (6 features) vs v2 (24 features)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def check_overfitting(ticker="TSLA", prediction_days=5):
    # Download data
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # ========== v1 - 6 features ==========
    df_v1 = df.copy()
    df_v1['MA_5'] = df_v1['Close'].rolling(5).mean()
    df_v1['MA_20'] = df_v1['Close'].rolling(20).mean()
    df_v1['Lag_1'] = df_v1['Close'].shift(1)
    df_v1['Lag_2'] = df_v1['Close'].shift(2)
    df_v1['Lag_3'] = df_v1['Close'].shift(3)
    df_v1['Volatility'] = df_v1['Close'].pct_change().rolling(20).std()
    df_v1['Target'] = df_v1['Close'].shift(-prediction_days)
    df_v1 = df_v1.dropna()
    
    features_v1 = ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility']
    X_v1 = df_v1[features_v1]
    y_v1 = df_v1['Target']
    
    # ========== v2 - 24 features ==========
    df_v2 = df.copy()
    # Base
    df_v2['MA_5'] = df_v2['Close'].rolling(5).mean()
    df_v2['MA_20'] = df_v2['Close'].rolling(20).mean()
    df_v2['Lag_1'] = df_v2['Close'].shift(1)
    df_v2['Lag_2'] = df_v2['Close'].shift(2)
    df_v2['Lag_3'] = df_v2['Close'].shift(3)
    df_v2['Volatility'] = df_v2['Close'].pct_change().rolling(20).std()
    
    # Bollinger Bands
    ma20 = df_v2['Close'].rolling(20).mean()
    std20 = df_v2['Close'].rolling(20).std()
    df_v2['BB_upper'] = ma20 + 2 * std20
    df_v2['BB_lower'] = ma20 - 2 * std20
    df_v2['BB_width'] = df_v2['BB_upper'] - df_v2['BB_lower']
    df_v2['BB_position'] = (df_v2['Close'] - df_v2['BB_lower']) / (df_v2['BB_upper'] - df_v2['BB_lower'])
    
    # Price Changes
    df_v2['Price_change_1d'] = df_v2['Close'].diff(1)
    df_v2['Price_change_3d'] = df_v2['Close'].diff(3)
    df_v2['Price_pct_1d'] = df_v2['Close'].pct_change(1)
    df_v2['Price_pct_3d'] = df_v2['Close'].pct_change(3)
    
    # RoC
    df_v2['RoC_5'] = df_v2['Close'].pct_change(5)
    df_v2['RoC_10'] = df_v2['Close'].pct_change(10)
    
    # Momentum
    df_v2['Momentum_5'] = df_v2['Close'] - df_v2['Close'].shift(5)
    df_v2['Momentum_10'] = df_v2['Close'] - df_v2['Close'].shift(10)
    
    # MACD
    df_v2['EMA_12'] = df_v2['Close'].ewm(span=12, adjust=False).mean()
    df_v2['EMA_26'] = df_v2['Close'].ewm(span=26, adjust=False).mean()
    df_v2['MACD'] = df_v2['EMA_12'] - df_v2['EMA_26']
    df_v2['MACD_signal'] = df_v2['MACD'].ewm(span=9, adjust=False).mean()
    df_v2['MACD_hist'] = df_v2['MACD'] - df_v2['MACD_signal']
    
    # ATR
    df_v2['H-L'] = df_v2['High'] - df_v2['Low']
    df_v2['H-PC'] = (df_v2['High'] - df_v2['Close'].shift(1)).abs()
    df_v2['L-PC'] = (df_v2['Low'] - df_v2['Close'].shift(1)).abs()
    df_v2['TR'] = df_v2[['H-L','H-PC','L-PC']].max(axis=1)
    df_v2['ATR'] = df_v2['TR'].rolling(14).mean()
    
    # Volume
    df_v2['Volume_MA_5'] = df_v2['Volume'].rolling(5).mean()
    df_v2['Volume_Ratio'] = df_v2['Volume'] / df_v2['Volume_MA_5']
    
    df_v2['Target'] = df_v2['Close'].shift(-prediction_days)
    df_v2 = df_v2.dropna()
    
    features_v2 = [
        'MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility',
        'BB_width', 'BB_position',
        'Price_change_1d', 'Price_change_3d', 'Price_pct_1d', 'Price_pct_3d',
        'RoC_5', 'RoC_10',
        'Momentum_5', 'Momentum_10',
        'MACD', 'MACD_signal', 'MACD_hist',
        'ATR',
        'Volume_MA_5', 'Volume_Ratio',
        'EMA_12', 'EMA_26'
    ]
    X_v2 = df_v2[features_v2]
    y_v2 = df_v2['Target']
    
    # ========== Train/Test Split (same for both) ==========
    split_v1 = int(len(X_v1) * 0.8)
    split_v2 = int(len(X_v2) * 0.8)
    
    X_train_v1, X_test_v1 = X_v1[:split_v1], X_v1[split_v1:]
    y_train_v1, y_test_v1 = y_v1[:split_v1], y_v1[split_v1:]
    
    X_train_v2, X_test_v2 = X_v2[:split_v2], X_v2[split_v2:]
    y_train_v2, y_test_v2 = y_v2[:split_v2], y_v2[split_v2:]
    
    # ========== SAME hyperparameters for both ==========
    hyperparams = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'verbosity': 0
    }
    
    # Train v1
    model_v1 = XGBRegressor(**hyperparams)
    model_v1.fit(X_train_v1, y_train_v1)
    
    pred_train_v1 = model_v1.predict(X_train_v1)
    pred_test_v1 = model_v1.predict(X_test_v1)
    
    r2_train_v1 = r2_score(y_train_v1, pred_train_v1)
    r2_test_v1 = r2_score(y_test_v1, pred_test_v1)
    rmse_train_v1 = np.sqrt(mean_squared_error(y_train_v1, pred_train_v1))
    rmse_test_v1 = np.sqrt(mean_squared_error(y_test_v1, pred_test_v1))
    
    # Train v2
    model_v2 = XGBRegressor(**hyperparams)
    model_v2.fit(X_train_v2, y_train_v2)
    
    pred_train_v2 = model_v2.predict(X_train_v2)
    pred_test_v2 = model_v2.predict(X_test_v2)
    
    r2_train_v2 = r2_score(y_train_v2, pred_train_v2)
    r2_test_v2 = r2_score(y_test_v2, pred_test_v2)
    rmse_train_v2 = np.sqrt(mean_squared_error(y_train_v2, pred_train_v2))
    rmse_test_v2 = np.sqrt(mean_squared_error(y_test_v2, pred_test_v2))
    
    # ========== RESULTS ==========
    print("\n" + "="*80)
    print(f"OVERFITTING CHECK - {ticker}")
    print("="*80)
    print(f"Hyperparameters: {hyperparams}")
    print("="*80)
    
    print("\n📊 XGBoost v1 (6 features)")
    print("-"*80)
    print(f"TRAIN: R²={r2_train_v1:.4f} | RMSE=${rmse_train_v1:.2f}")
    print(f"TEST:  R²={r2_test_v1:.4f} | RMSE=${rmse_test_v1:.2f}")
    gap_r2_v1 = r2_train_v1 - r2_test_v1
    gap_rmse_v1 = rmse_test_v1 - rmse_train_v1
    print(f"GAP:   ΔR²={gap_r2_v1:.4f} | ΔRMSE=${gap_rmse_v1:.2f}")
    
    print("\n📊 XGBoost v2 (24 features)")
    print("-"*80)
    print(f"TRAIN: R²={r2_train_v2:.4f} | RMSE=${rmse_train_v2:.2f}")
    print(f"TEST:  R²={r2_test_v2:.4f} | RMSE=${rmse_test_v2:.2f}")
    gap_r2_v2 = r2_train_v2 - r2_test_v2
    gap_rmse_v2 = rmse_test_v2 - rmse_train_v2
    print(f"GAP:   ΔR²={gap_r2_v2:.4f} | ΔRMSE=${gap_rmse_v2:.2f}")
    
    # ========== OVERFITTING ANALYSIS ==========
    print("\n" + "="*80)
    print("OVERFITTING DIAGNOSTIC")
    print("="*80)
    
    # R² gap analysis
    print(f"\nR² Train-Test Gap:")
    print(f"  v1 (6 features):  {gap_r2_v1:.4f}")
    print(f"  v2 (24 features): {gap_r2_v2:.4f}")
    
    if gap_r2_v2 > gap_r2_v1:
        print(f"  ⚠️  v2 has LARGER gap (+{gap_r2_v2 - gap_r2_v1:.4f}) → More overfitting")
    else:
        print(f"  ✅ v2 has SMALLER gap ({gap_r2_v2 - gap_r2_v1:.4f}) → Less overfitting")
    
    # RMSE gap analysis
    print(f"\nRMSE Train-Test Gap:")
    print(f"  v1 (6 features):  ${gap_rmse_v1:.2f}")
    print(f"  v2 (24 features): ${gap_rmse_v2:.2f}")
    
    if gap_rmse_v2 > gap_rmse_v1:
        print(f"  ⚠️  v2 has LARGER gap (+${gap_rmse_v2 - gap_rmse_v1:.2f}) → More overfitting")
    else:
        print(f"  ✅ v2 has SMALLER gap (${gap_rmse_v2 - gap_rmse_v1:.2f}) → Less overfitting")
    
    # Overfitting severity
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    # Rules of thumb for overfitting
    overfitting_threshold_r2 = 0.10  # >10% gap = concerning
    overfitting_threshold_rmse_pct = 0.15  # >15% increase = concerning
    
    v1_overfitting = "None"
    if gap_r2_v1 > overfitting_threshold_r2:
        v1_overfitting = "Severe"
    elif gap_r2_v1 > 0.05:
        v1_overfitting = "Moderate"
    elif gap_r2_v1 > 0.02:
        v1_overfitting = "Slight"
    
    v2_overfitting = "None"
    if gap_r2_v2 > overfitting_threshold_r2:
        v2_overfitting = "Severe"
    elif gap_r2_v2 > 0.05:
        v2_overfitting = "Moderate"
    elif gap_r2_v2 > 0.02:
        v2_overfitting = "Slight"
    
    print(f"\nv1 (6 features):  {v1_overfitting} overfitting (ΔR²={gap_r2_v1:.4f})")
    print(f"v2 (24 features): {v2_overfitting} overfitting (ΔR²={gap_r2_v2:.4f})")
    
    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if gap_r2_v2 < 0.05 and r2_test_v2 > r2_test_v1:
        print("✅ v2 is SAFE to use:")
        print(f"   - Minimal overfitting (ΔR²={gap_r2_v2:.4f} < 0.05)")
        print(f"   - Better test performance (R²={r2_test_v2:.4f} vs {r2_test_v1:.4f})")
        print(f"   - Improvement: +{(r2_test_v2 - r2_test_v1):.4f} R² on unseen data")
    elif gap_r2_v2 > gap_r2_v1 and gap_r2_v2 > 0.05:
        print("⚠️  v2 shows signs of overfitting:")
        print(f"   - Train-test gap increased: {gap_r2_v2:.4f} vs {gap_r2_v1:.4f}")
        print("   - Consider: Reduce features, increase regularization, or use cross-validation")
    else:
        print("✅ Both models are reasonably well-generalized")
        print(f"   - v1 gap: {gap_r2_v1:.4f}")
        print(f"   - v2 gap: {gap_r2_v2:.4f}")
        print(f"   - v2 better test R²: {r2_test_v2:.4f} vs {r2_test_v1:.4f}")
    
    print("="*80 + "\n")
    
    return {
        'v1': {'train_r2': r2_train_v1, 'test_r2': r2_test_v1, 'gap_r2': gap_r2_v1,
               'train_rmse': rmse_train_v1, 'test_rmse': rmse_test_v1, 'gap_rmse': gap_rmse_v1},
        'v2': {'train_r2': r2_train_v2, 'test_r2': r2_test_v2, 'gap_r2': gap_r2_v2,
               'train_rmse': rmse_train_v2, 'test_rmse': rmse_test_v2, 'gap_rmse': gap_rmse_v2}
    }


if __name__ == "__main__":
    results = check_overfitting(ticker="TSLA", prediction_days=5)
