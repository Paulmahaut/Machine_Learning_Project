"""
Reduce Overfitting - Test different regularization strategies
Goal: Reduce train-test gap while maintaining good test performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def prepare_data_v2(ticker="TSLA", prediction_days=5):
    """Prepare data with 24 features (v2)"""
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Base features
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # Bollinger Bands
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Price Changes
    df['Price_change_1d'] = df['Close'].diff(1)
    df['Price_change_3d'] = df['Close'].diff(3)
    df['Price_pct_1d'] = df['Close'].pct_change(1)
    df['Price_pct_3d'] = df['Close'].pct_change(3)
    
    # RoC
    df['RoC_5'] = df['Close'].pct_change(5)
    df['RoC_10'] = df['Close'].pct_change(10)
    
    # Momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Volume
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    df['Target'] = df['Close'].shift(-prediction_days)
    df = df.dropna()
    
    features = [
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
    
    return df[features], df['Target']


def test_configuration(X, y, config_name, **params):
    """Test a configuration and return metrics"""
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = XGBRegressor(**params, verbosity=0)
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    gap = r2_train - r2_test
    
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    
    return {
        'name': config_name,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'gap': gap,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'params': params
    }


print("\n" + "="*90)
print("REDUCING OVERFITTING - Testing Multiple Strategies")
print("="*90)

# Prepare data
X, y = prepare_data_v2("TSLA", prediction_days=5)

# Test configurations
configs = []

# 1. Baseline (current v2)
print("\n1️⃣  Baseline (current v2)")
configs.append(test_configuration(X, y, "Baseline v2",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
))

# 2. Reduce max_depth
print("2️⃣  Reduce max_depth (5 → 3)")
configs.append(test_configuration(X, y, "max_depth=3",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
))

# 3. Reduce learning_rate
print("3️⃣  Reduce learning_rate (0.1 → 0.05)")
configs.append(test_configuration(X, y, "lr=0.05",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05
))

# 4. Add L2 regularization (lambda)
print("4️⃣  L2 regularization (reg_lambda=1)")
configs.append(test_configuration(X, y, "L2 reg_lambda=1",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    reg_lambda=1
))

# 5. Add L1 regularization (alpha)
print("5️⃣  L1 regularization (reg_alpha=0.5)")
configs.append(test_configuration(X, y, "L1 reg_alpha=0.5",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    reg_alpha=0.5
))

# 6. Increase min_child_weight
print("6️⃣  Increase min_child_weight (1 → 5)")
configs.append(test_configuration(X, y, "min_child_weight=5",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_child_weight=5
))

# 7. Add subsample (row sampling)
print("7️⃣  Subsample rows (80%)")
configs.append(test_configuration(X, y, "subsample=0.8",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8
))

# 8. Add colsample_bytree (feature sampling)
print("8️⃣  Subsample features (80%)")
configs.append(test_configuration(X, y, "colsample=0.8",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=0.8
))

# 9. Combo: depth + regularization
print("9️⃣  Combo (depth=3 + L2=1)")
configs.append(test_configuration(X, y, "Combo depth+L2",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    reg_lambda=1
))

# 10. Combo: Best of everything
print("🔟 Best Combo (depth=3 + L2=1 + subsample)")
configs.append(test_configuration(X, y, "Best Combo",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    reg_lambda=1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3
))

# Results table
print("\n" + "="*90)
print("RESULTS COMPARISON")
print("="*90)
print(f"{'Configuration':<20} {'R² Train':>10} {'R² Test':>10} {'Gap':>10} {'RMSE Test':>12}")
print("-"*90)

for config in configs:
    print(f"{config['name']:<20} {config['r2_train']:>10.4f} {config['r2_test']:>10.4f} "
          f"{config['gap']:>10.4f} ${config['rmse_test']:>11.2f}")

# Find best configuration (minimize gap while maintaining good test R²)
print("\n" + "="*90)
print("ANALYSIS")
print("="*90)

# Sort by gap (ascending)
sorted_by_gap = sorted(configs, key=lambda x: x['gap'])
print("\n🏆 Best Gap Reduction:")
for i, config in enumerate(sorted_by_gap[:3], 1):
    print(f"{i}. {config['name']:<20} Gap={config['gap']:.4f} | R² test={config['r2_test']:.4f}")

# Sort by test R² (descending)
sorted_by_test = sorted(configs, key=lambda x: x['r2_test'], reverse=True)
print("\n🎯 Best Test Performance:")
for i, config in enumerate(sorted_by_test[:3], 1):
    print(f"{i}. {config['name']:<20} R² test={config['r2_test']:.4f} | Gap={config['gap']:.4f}")

# Find sweet spot (good gap AND good test performance)
print("\n💎 Sweet Spot (Gap < 0.15 AND R² test > 0.80):")
sweet_spots = [c for c in configs if c['gap'] < 0.15 and c['r2_test'] > 0.80]
if sweet_spots:
    for config in sorted(sweet_spots, key=lambda x: x['r2_test'], reverse=True):
        print(f"✅ {config['name']:<20} R² test={config['r2_test']:.4f} | Gap={config['gap']:.4f}")
else:
    print("⚠️  No config meets both criteria. Relaxing...")
    sweet_spots = [c for c in configs if c['gap'] < 0.18 and c['r2_test'] > 0.78]
    for config in sorted(sweet_spots, key=lambda x: x['r2_test'], reverse=True):
        print(f"⚠️  {config['name']:<20} R² test={config['r2_test']:.4f} | Gap={config['gap']:.4f}")

# Recommendation
print("\n" + "="*90)
print("RECOMMENDATION")
print("="*90)

baseline = configs[0]
best_combo = configs[-1]

print(f"\nBaseline (v2):")
print(f"  R² train={baseline['r2_train']:.4f} | R² test={baseline['r2_test']:.4f} | Gap={baseline['gap']:.4f}")

print(f"\nBest Combo:")
print(f"  R² train={best_combo['r2_train']:.4f} | R² test={best_combo['r2_test']:.4f} | Gap={best_combo['gap']:.4f}")

improvement_gap = baseline['gap'] - best_combo['gap']
improvement_test = best_combo['r2_test'] - baseline['r2_test']

print(f"\nImprovements:")
print(f"  Gap reduction: {improvement_gap:+.4f} ({improvement_gap/baseline['gap']*100:+.1f}%)")
print(f"  R² test change: {improvement_test:+.4f} ({improvement_test/baseline['r2_test']*100:+.1f}%)")

if best_combo['gap'] < 0.15 and best_combo['r2_test'] > 0.78:
    print(f"\n✅ RECOMMENDED: Use '{best_combo['name']}' configuration")
    print(f"   Gap < 0.15 ({best_combo['gap']:.4f}) AND R² test > 0.78 ({best_combo['r2_test']:.4f})")
    print(f"\n   Parameters:")
    for key, value in best_combo['params'].items():
        print(f"     {key}: {value}")
else:
    # Find best alternative
    best_alt = max([c for c in configs if c['gap'] < baseline['gap']], 
                   key=lambda x: x['r2_test'])
    print(f"\n✅ RECOMMENDED: Use '{best_alt['name']}' configuration")
    print(f"   Best balance: Gap={best_alt['gap']:.4f} | R² test={best_alt['r2_test']:.4f}")
    print(f"\n   Parameters:")
    for key, value in best_alt['params'].items():
        print(f"     {key}: {value}")

print("\n" + "="*90 + "\n")
