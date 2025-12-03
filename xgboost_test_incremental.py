"""
XGBoost Incremental Testing - Test progressif des features
Basé sur xgboost_simple.py (qui fonctionne) + ajout d'EMA
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def run_xgboost_test(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True):
    """
    Test avec les features de base + EMA (Exponential Moving Average)
    """
    
    # Download data
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"\n{'='*80}")
    print(f"📊 TEST INCRÉMENTAL - {name} ({ticker})")
    print(f"{'='*80}")
    print(f"Dataset: {len(df)} jours de données")
    print(f"Horizon de prédiction: {prediction_days} jour(s)")
    
    # ========== FEATURES DE BASE (xgboost_simple.py) ==========
    # Moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    
    # Price lags (historical prices)
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # ========== NOUVELLES FEATURES À TESTER (de xgBoostEval2) ==========
    # EMA (Exponential Moving Average) - Plus réactif que MA classique
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Target variable (price to predict)
    df['Target'] = df['Close'].shift(-prediction_days)
    df = df.dropna()
    
    print(f"Données après nettoyage: {len(df)} observations")
    
    # ========== TEST 1: FEATURES DE BASE SEULEMENT ==========
    features_base = ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility']
    
    X_base = df[features_base]
    y = df['Target']
    
    split = int(len(X_base) * 0.8)
    X_train_base = X_base[:split]
    X_test_base = X_base[split:]
    y_train = y[:split]
    y_test = y[split:]
    
    print(f"\nTrain: {len(X_train_base)} observations | Test: {len(X_test_base)} observations")
    
    # Modèle de base
    model_base = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
    model_base.fit(X_train_base, y_train)
    preds_base = model_base.predict(X_test_base)
    
    rmse_base = np.sqrt(mean_squared_error(y_test, preds_base))
    mae_base = mean_absolute_error(y_test, preds_base)
    r2_base = r2_score(y_test, preds_base)
    
    print(f"\n{'─'*80}")
    print(f"🔵 MODÈLE DE BASE (6 features: MA_5, MA_20, Lag_1-3, Volatility)")
    print(f"{'─'*80}")
    print(f"   R²   = {r2_base:.4f}")
    print(f"   RMSE = ${rmse_base:.2f}")
    print(f"   MAE  = ${mae_base:.2f}")
    
    # ========== TEST 2: FEATURES DE BASE + EMA ==========
    features_extended = features_base + ['EMA_12', 'EMA_26']
    
    X_extended = df[features_extended]
    X_train_ext = X_extended[:split]
    X_test_ext = X_extended[split:]
    
    # Modèle avec EMA
    model_ext = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
    model_ext.fit(X_train_ext, y_train)
    preds_ext = model_ext.predict(X_test_ext)
    
    rmse_ext = np.sqrt(mean_squared_error(y_test, preds_ext))
    mae_ext = mean_absolute_error(y_test, preds_ext)
    r2_ext = r2_score(y_test, preds_ext)
    
    print(f"\n{'─'*80}")
    print(f"🟢 MODÈLE ÉTENDU (8 features: BASE + EMA_12, EMA_26)")
    print(f"{'─'*80}")
    print(f"   R²   = {r2_ext:.4f}")
    print(f"   RMSE = ${rmse_ext:.2f}")
    print(f"   MAE  = ${mae_ext:.2f}")
    
    # ========== COMPARAISON ==========
    r2_diff = r2_ext - r2_base
    rmse_diff = rmse_ext - rmse_base
    mae_diff = mae_ext - mae_base
    
    print(f"\n{'='*80}")
    print(f"📈 IMPACT DE L'AJOUT DE EMA")
    print(f"{'='*80}")
    print(f"   ΔR²   = {r2_diff:+.4f}  {'✅ AMÉLIORATION' if r2_diff > 0 else '❌ DÉGRADATION'}")
    print(f"   ΔRMSE = ${rmse_diff:+.2f}  {'✅ AMÉLIORATION' if rmse_diff < 0 else '❌ DÉGRADATION'}")
    print(f"   ΔMAE  = ${mae_diff:+.2f}  {'✅ AMÉLIORATION' if mae_diff < 0 else '❌ DÉGRADATION'}")
    
    # Verdict
    print(f"\n{'='*80}")
    if r2_diff > 0.01 or (rmse_diff < 0 and mae_diff < 0):
        print("✅ VERDICT: EMA améliore les performances → À CONSERVER")
        recommendation = "CONSERVER"
    elif abs(r2_diff) < 0.01 and abs(rmse_diff) < 0.5:
        print("⚠️  VERDICT: EMA n'a pas d'impact significatif → Neutre")
        recommendation = "NEUTRE"
    else:
        print("❌ VERDICT: EMA dégrade les performances → À SUPPRIMER")
        recommendation = "SUPPRIMER"
    print(f"{'='*80}\n")
    
    # Feature importance
    importances_ext = model_ext.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': features_extended,
        'Importance': importances_ext
    }).sort_values('Importance', ascending=False)
    
    print("🎯 IMPORTANCE DES FEATURES (modèle étendu):")
    print(feature_imp_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Prédictions vs Réel
    axes[0].plot(y_test.values, label='Prix Réel', alpha=0.7, linewidth=2)
    axes[0].plot(preds_base, label='Prédictions BASE', alpha=0.6, linestyle='--')
    axes[0].plot(preds_ext, label='Prédictions BASE+EMA', alpha=0.6, linestyle='--')
    axes[0].set_title(f'{name} - Prédictions vs Prix Réel', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Jours (test set)')
    axes[0].set_ylabel('Prix ($)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Feature Importance
    axes[1].barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='skyblue')
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Importance des Features (BASE+EMA)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'test_incremental_{ticker}.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé: test_incremental_{ticker}.png")
    
    return {
        'base': {'r2': r2_base, 'rmse': rmse_base, 'mae': mae_base},
        'extended': {'r2': r2_ext, 'rmse': rmse_ext, 'mae': mae_ext},
        'recommendation': recommendation,
        'feature_importance': feature_imp_df
    }


if __name__ == "__main__":
    # Test sur TotalEnergies
    print("\n" + "🚀 DÉBUT DU TEST INCRÉMENTAL" + "\n")
    results = run_xgboost_test(ticker="TTE.PA", name="TotalEnergies", prediction_days=5, verbose=True)
    
    print(f"\n{'='*80}")
    print("🏁 TEST TERMINÉ")
    print(f"{'='*80}")
    print(f"Recommandation finale: {results['recommendation']}")
    print(f"{'='*80}\n")
