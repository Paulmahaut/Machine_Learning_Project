"""
Quick check: Are all 24 features really important?
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Download data
df = yf.download("TSLA", start="2015-01-01", end="2025-01-01", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# All features from xgboost_simple2.py
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df['Volatility'] = df['Close'].pct_change().rolling(20).std()

ma20 = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BB_upper'] = ma20 + 2 * std20
df['BB_lower'] = ma20 - 2 * std20
df['BB_width'] = df['BB_upper'] - df['BB_lower']
df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

df['Price_change_1d'] = df['Close'].diff(1)
df['Price_change_3d'] = df['Close'].diff(3)
df['Price_pct_1d'] = df['Close'].pct_change(1)
df['Price_pct_3d'] = df['Close'].pct_change(3)

df['RoC_5'] = df['Close'].pct_change(5)
df['RoC_10'] = df['Close'].pct_change(10)

df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

df['H-L'] = df['High'] - df['Low']
df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
df['ATR'] = df['TR'].rolling(14).mean()

df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']

df['Target'] = df['Close'].shift(-5)
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

X = df[features]
y = df['Target']

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
model.fit(X_train, y_train)

# Feature importance analysis
importances = model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS - All 24 Features")
print("="*70)
print(feature_imp_df.to_string(index=False))

# Calculate cumulative importance
feature_imp_df['Cumulative'] = feature_imp_df['Importance'].cumsum()

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)

# Top features
top3 = feature_imp_df.head(3)
top3_total = top3['Importance'].sum()
print(f"\nTop 3 features represent {top3_total*100:.2f}% of total importance:")
for idx, row in top3.iterrows():
    print(f"  - {row['Feature']}: {row['Importance']*100:.2f}%")

top10 = feature_imp_df.head(10)
top10_total = top10['Importance'].sum()
print(f"\nTop 10 features represent {top10_total*100:.2f}% of total importance")

# Near-zero importance
threshold = 0.001
low_importance = feature_imp_df[feature_imp_df['Importance'] < threshold]
print(f"\n{len(low_importance)} features have importance < 0.1%:")
if len(low_importance) > 0:
    for idx, row in low_importance.iterrows():
        print(f"  - {row['Feature']}: {row['Importance']*100:.4f}%")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
if len(low_importance) > 5:
    print(f"⚠️  {len(low_importance)} features contribute very little to the model")
    print(f"💡 Consider keeping only the top {24 - len(low_importance)} most important features")
    print(f"   This would simplify the model without losing performance")
else:
    print("✅ Most features contribute meaningfully to the model")
    print("   The 24 features are justified")

print("\n" + "="*70 + "\n")
