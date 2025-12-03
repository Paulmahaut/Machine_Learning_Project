"""
COMPARISON: Eval2 vs AlexisEval2 on TSLA
"""

print("\n" + "="*90)
print("📊 COMPARISON: xgboost_simple (Eval2) vs xgboost_simple2 (AlexisEval2) on TESLA")
print("="*90 + "\n")

print("Test Conditions:")
print("  • Asset: TSLA (Tesla)")
print("  • Dataset: 2015-2025")
print("  • Split: 80/20 (Train: 1992 days | Test: 499 days)")
print("  • Prediction horizon: 5 days")
print("  • Model: XGBoost (n_estimators=100, max_depth=5, lr=0.1)")

print("\n" + "-"*90)
print("MODEL 1: xgboost_simple (Eval2 branch)")
print("-"*90)
print("Features: 6 (base features only)")
print("  ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility']")
print("\nPerformance on TSLA:")
print("  R²   = 0.7874  ✅ EXCELLENT")
print("  RMSE = $26.80")
print("  MAE  = N/A (not computed in Eval2)")

print("\n" + "-"*90)
print("MODEL 2: xgboost_simple2 (AlexisEval2 branch)")
print("-"*90)
print("Features: 14 (6 base + 8 optimized additions)")
print("  Base (6): ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility']")
print("  + Rolling Stats (5): ['Rolling_std_20', 'Rolling_min_20', 'Rolling_max_20',")
print("                        'Rolling_skew_20', 'Rolling_kurt_20']")
print("  + RSI (1): ['RSI']")
print("  + Volatility (2): ['Volatility_5', 'Volatility_20_alt']")
print("\nPerformance on TSLA:")
print("  R²   = 0.7417  ✅ VERY GOOD")
print("  RMSE = $29.53")
print("  MAE  = $21.02")

print("\n" + "="*90)
print("📈 ANALYSIS")
print("="*90)

r2_eval2 = 0.7874
r2_alexis = 0.7417
rmse_eval2 = 26.80
rmse_alexis = 29.53

r2_diff = r2_alexis - r2_eval2
rmse_diff = rmse_alexis - rmse_eval2
rmse_diff_pct = (rmse_diff / rmse_eval2) * 100

print(f"\nΔR²   = {r2_diff:+.4f}  ❌ SLIGHT DEGRADATION")
print(f"ΔRMSE = ${rmse_diff:+.2f} ({rmse_diff_pct:+.1f}%)  ❌ 10.2% worse")

print("\n" + "="*90)
print("🎯 VERDICT")
print("="*90)

print("\n❌ For TESLA: xgboost_simple (Eval2) performs BETTER")
print("\nReasons:")
print("  1. Higher R² (0.787 vs 0.742) → -0.045 difference")
print("  2. Lower RMSE ($26.80 vs $29.53) → $2.73 worse")
print("  3. Simpler model (6 features vs 14) → Less overfitting risk")
print("  4. The additional 8 features don't help for TSLA")

print("\n💡 KEY INSIGHT:")
print("  • The 'optimized' features (Rolling Stats, extra Volatility) were selected")
print("    based on TotalEnergies (TTE.PA) testing")
print("  • What works for one asset doesn't necessarily work for another!")
print("  • TSLA is more predictable with simple features (Lag_1 + MA_5)")
print("  • TTE.PA needs more complex features but still performs poorly")

print("\n📊 FEATURE IMPORTANCE COMPARISON:")
print("\n  Eval2 (implicit from R²):")
print("    → Likely dominated by Lag_1 and MA_5 (like simple2)")
print("\n  AlexisEval2 (xgboost_simple2):")
print("    1. Lag_1 (38.2%)")
print("    2. Rolling_max_20 (32.5%)")
print("    3. Rolling_min_20 (15.1%)")
print("    4. MA_5 (13.3%)")
print("    → Additional features add noise, reducing performance")

print("\n" + "="*90)
print("✅ FINAL ANSWER: Eval2 (simple 6 features) is BETTER for TSLA")
print("="*90)

print("\n🔬 RECOMMENDATION:")
print("  • For TSLA/US tech stocks: Use Eval2 (6 base features)")
print("  • For harder assets (TTE.PA): xgboost_simple2 reduces the loss")
print("  • Or: Make feature selection ASSET-SPECIFIC!")
print("="*90 + "\n")
