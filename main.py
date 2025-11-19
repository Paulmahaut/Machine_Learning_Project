"""
ML Project - Prophet vs XGBoost Comparison
"""

from prophet_algo import run_prophet
from xgboost_simple import run_xgboost
import pandas as pd

# Configuration
TICKER_FOREX = "EURUSD=X"
NAME_FOREX = "EUR/USD"
TICKER_STOCK = "TSLA"
NAME_STOCK = "Tesla"

print("\nML Project - Regression Algorithms Comparison")
print("="*60)

# Test Prophet
print("\n[1/2] Prophet on EUR/USD...")
try:
    prophet_results = run_prophet(TICKER_FOREX, NAME_FOREX, test_years=3, verbose=False)
    prophet_success = True
except Exception as e:
    print("Prophet failed (CmdStan Windows error)")
    prophet_success = False
    prophet_results = None

# Test XGBoost
print("\n[2/2] XGBoost on Tesla...")
xgboost_results = run_xgboost(TICKER_STOCK, NAME_STOCK, prediction_days=5, verbose=False)

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

if prophet_success:
    print(f"\nProphet (EUR/USD):  R²={prophet_results['metrics']['R²']:.4f}  RMSE={prophet_results['metrics']['RMSE']:.6f}")
    print(f"XGBoost (Tesla):    R²={xgboost_results['metrics']['R²']:.4f}  RMSE=${xgboost_results['metrics']['RMSE']:.2f}")
else:
    print(f"\nXGBoost (Tesla):    R²={xgboost_results['metrics']['R²']:.4f}  RMSE=${xgboost_results['metrics']['RMSE']:.2f}")
    print("Prophet: Failed (Windows CmdStan error)")

print("\nAnalysis complete. Check generated plots.")
