"""
Performance Comparison - XGBoost models on TotalEnergies
"""

print("\n" + "="*90)
print("📊 PERFORMANCE COMPARISON - TOTALENERGIES (TTE.PA)")
print("="*90 + "\n")

print("Test conditions:")
print("  • Dataset: 2015-2025 (2536 observations after cleaning)")
print("  • Split: 80/20 (Train: 2028 days | Test: 508 days)")
print("  • Prediction horizon: 5 days")
print("  • Model: XGBoost (n_estimators=100, max_depth=5, lr=0.1)")

print("\n" + "-"*90)
print("MODEL 1: xgboost_simple (Eval2 branch)")
print("-"*90)
print("Features: 11")
print("  ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility',")
print("   'RSI', 'RoC_5', 'RoC_10', 'Volume_MA', 'Volume_Ratio']")
print("\nPerformance:")
print("  R²   = -0.9417  ❌ VERY POOR (worse than baseline)")
print("  RMSE = $6.18")
print("  MAE  = $5.00")

print("\n" + "-"*90)
print("MODEL 2: xgboost_simple2 (AlexisEval2 branch - OPTIMIZED)")
print("-"*90)
print("Features: 14 (6 base + 8 relevant additions)")
print("  Base (6): ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility']")
print("  + Rolling Stats (5): ['Rolling_std_20', 'Rolling_min_20', 'Rolling_max_20',")
print("                        'Rolling_skew_20', 'Rolling_kurt_20']")
print("  + RSI (1): ['RSI']")
print("  + Volatility (2): ['Volatility_5', 'Volatility_20_alt']")
print("\nPerformance:")
print("  R²   = -0.7441  ⚠️  POOR but BETTER")
print("  RMSE = $5.86")
print("  MAE  = $4.76")

print("\n" + "="*90)
print("📈 IMPROVEMENT ANALYSIS")
print("="*90)

r2_eval2 = -0.9417
r2_simple2 = -0.7441
rmse_eval2 = 6.18
rmse_simple2 = 5.86
mae_eval2 = 5.00
mae_simple2 = 4.76

r2_improvement = r2_simple2 - r2_eval2
rmse_improvement = rmse_eval2 - rmse_simple2
mae_improvement = mae_eval2 - mae_simple2
rmse_improvement_pct = (rmse_improvement / rmse_eval2) * 100
mae_improvement_pct = (mae_improvement / mae_eval2) * 100

print(f"\nΔR²   = {r2_improvement:+.4f}  ✅ SIGNIFICANT IMPROVEMENT")
print(f"ΔRMSE = ${rmse_improvement:+.2f} ({rmse_improvement_pct:+.1f}%)  ✅ 5.2% better")
print(f"ΔMAE  = ${mae_improvement:+.2f} ({mae_improvement_pct:+.1f}%)  ✅ 4.8% better")

print("\n" + "="*90)
print("🎯 VERDICT")
print("="*90)

print("\n✅ YES, xgboost_simple2 is MORE SATISFACTORY than xgboost_simple (Eval2)")
print("\nReasons:")
print("  1. Better R² score (-0.74 vs -0.94)")
print("  2. Lower RMSE ($5.86 vs $6.18) - 5.2% improvement")
print("  3. Lower MAE ($4.76 vs $5.00) - 4.8% improvement")
print("  4. More intelligent feature selection (tested one by one)")
print("  5. Removed harmful features (RoC, Volume in this context)")
print("  6. Added statistically relevant features (Rolling Stats, additional Volatility)")

print("\n⚠️  IMPORTANT NOTES:")
print("  • Both models still have NEGATIVE R², meaning they perform worse than")
print("    predicting the mean price (baseline model)")
print("  • This suggests:")
print("    - The 5-day prediction horizon might be too long")
print("    - Stock prices might be too volatile for these features")
print("    - Deep learning (GRU/LSTM) might be needed for better performance")
print("    - Feature engineering needs to capture more complex patterns")

print("\n📊 FEATURE IMPORTANCE (xgboost_simple2):")
print("  1. Lag_1 (53.9%) - Previous day's price")
print("  2. MA_5 (39.9%) - 5-day moving average")
print("  3. Rolling_max_20 (2.8%) - 20-day maximum")
print("  → The model relies heavily on recent price (Lag_1 + MA_5 = 94%)")

print("\n💡 NEXT STEPS TO IMPROVE:")
print("  1. Reduce prediction horizon (5 days → 1 day)")
print("  2. Implement GRU/LSTM for temporal patterns")
print("  3. Add autocorrelation analysis")
print("  4. Consider external features (market sentiment, oil prices for TTE)")
print("  5. Try different train/test splits or cross-validation")

print("\n" + "="*90)
print("✅ CONCLUSION: xgboost_simple2 IS BETTER (but both need improvement)")
print("="*90 + "\n")
