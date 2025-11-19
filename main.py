"""
ML Project - Prophet vs XGBoost Comparison
"""

from prophet_algo import run_prophet
from xgboost_simple import run_xgboost

print("\nML Project - Regression Algorithms Comparison")
print("="*60)

# Test Prophet (EUR/USD forex)
print("\n[1/2] Prophet on EUR/USD...")
try:
    prophet_results = run_prophet()
    prophet_success = True
except Exception as e:
    print("Prophet failed (CmdStan Windows error)")
    prophet_success = False
    prophet_results = None

# Test XGBoost (Tesla)
print("\n[2/2] XGBoost on Tesla...")
xgboost_results = run_xgboost("TSLA", "Tesla", prediction_days=5, verbose=True)

# Results
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
