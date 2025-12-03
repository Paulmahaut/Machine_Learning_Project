"""
XGBoost Automatic Feature Testing - Test systématique de toutes les features
Compare le modèle de base avec l'ajout de chaque feature individuellement
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def get_base_features(df):
    """Features de base qui fonctionnent (de xgboost_simple.py)"""
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    return df


def test_single_feature(df, feature_name, feature_cols, y_train, y_test, split, base_features):
    """Test l'ajout d'une feature au modèle de base"""
    
    # Vérifier que toutes les colonnes existent
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        return None
    
    # Modèle avec feature ajoutée
    all_features = base_features + feature_cols
    X = df[all_features]
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Calculer l'importance de la nouvelle feature
    importances = model.feature_importances_
    new_feature_importance = sum(importances[len(base_features):])
    
    return {
        'name': feature_name,
        'columns': feature_cols,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'importance': new_feature_importance
    }


def run_comprehensive_test(ticker="TTE.PA", name="TotalEnergies", prediction_days=5):
    """Test toutes les features disponibles de xgBoostEval2"""
    
    print(f"\n{'='*90}")
    print(f"🧪 TEST COMPLET DE TOUTES LES FEATURES - {name}")
    print(f"{'='*90}\n")
    
    # Download data
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Features de base
    df = get_base_features(df)
    base_features = ['MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility']
    
    # ========== AJOUTER TOUTES LES FEATURES DE xgBoostEval2 ==========
    
    # 1. MA supplémentaire
    df['MA_50'] = df['Close'].rolling(50).mean()
    
    # 2. EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 3. MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 4. RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. Rate of Change
    df['RoC_5'] = df['Close'].pct_change(5)
    df['RoC_10'] = df['Close'].pct_change(10)
    
    # 6. Bollinger Bands
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # 7. ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # 8. Rolling Stats
    df['Rolling_std_20'] = df['Close'].rolling(20).std()
    df['Rolling_min_20'] = df['Close'].rolling(20).min()
    df['Rolling_max_20'] = df['Close'].rolling(20).max()
    df['Rolling_skew_20'] = df['Close'].rolling(20).skew()
    df['Rolling_kurt_20'] = df['Close'].rolling(20).kurt()
    
    # 9. Lags supplémentaires
    df['Lag_5'] = df['Close'].shift(5)
    df['Lag_10'] = df['Close'].shift(10)
    df['Lag_20'] = df['Close'].shift(20)
    
    # 10. Price Changes & Momentum
    df['Price_change_1d'] = df['Close'].diff(1)
    df['Price_change_3d'] = df['Close'].diff(3)
    df['Price_pct_1d'] = df['Close'].pct_change(1)
    df['Price_pct_3d'] = df['Close'].pct_change(3)
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # 11. Volatility supplémentaire
    df['Volatility_5'] = df['Close'].rolling(5).std()
    df['Volatility_20_alt'] = df['Close'].rolling(20).std()
    
    # 12. Volume
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    # 13. Temporal
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Weekday'] = df.index.weekday
    df['Is_month_end'] = df.index.is_month_end.astype(int)
    
    # Target
    df['Target'] = df['Close'].shift(-prediction_days)
    df = df.dropna()
    
    # Split
    split = int(len(df) * 0.8)
    y = df['Target']
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    
    # ========== BASELINE (modèle de base) ==========
    X_base = df[base_features]
    X_train_base = X_base.iloc[:split]
    X_test_base = X_base[split:]
    
    model_base = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
    model_base.fit(X_train_base, y_train)
    preds_base = model_base.predict(X_test_base)
    
    baseline_r2 = r2_score(y_test, preds_base)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, preds_base))
    baseline_mae = mean_absolute_error(y_test, preds_base)
    
    print(f"📊 BASELINE (6 features de base):")
    print(f"   R²   = {baseline_r2:.4f}")
    print(f"   RMSE = ${baseline_rmse:.2f}")
    print(f"   MAE  = ${baseline_mae:.2f}\n")
    
    # ========== TESTER CHAQUE FEATURE INDIVIDUELLEMENT ==========
    
    features_to_test = [
        ('MA_50', ['MA_50']),
        ('EMA (12, 26)', ['EMA_12', 'EMA_26']),
        ('MACD', ['MACD', 'MACD_signal', 'MACD_hist']),
        ('RSI', ['RSI']),
        ('RoC (5, 10)', ['RoC_5', 'RoC_10']),
        ('Bollinger Bands', ['BB_width', 'BB_position']),
        ('ATR', ['ATR']),
        ('Rolling Stats', ['Rolling_std_20', 'Rolling_min_20', 'Rolling_max_20', 'Rolling_skew_20', 'Rolling_kurt_20']),
        ('Lags (5, 10, 20)', ['Lag_5', 'Lag_10', 'Lag_20']),
        ('Price Changes', ['Price_change_1d', 'Price_change_3d', 'Price_pct_1d', 'Price_pct_3d']),
        ('Momentum (5, 10)', ['Momentum_5', 'Momentum_10']),
        ('Volatility (5, 20)', ['Volatility_5', 'Volatility_20_alt']),
        ('Volume', ['Volume_MA_5', 'Volume_Ratio']),
        ('Temporal', ['Day', 'Month', 'Weekday', 'Is_month_end'])
    ]
    
    results = []
    print(f"{'─'*90}")
    print(f"🧪 TEST DE CHAQUE FEATURE (BASE + feature):\n")
    
    for feature_name, feature_cols in features_to_test:
        result = test_single_feature(df, feature_name, feature_cols, y_train, y_test, split, base_features)
        
        if result:
            results.append(result)
            delta_r2 = result['r2'] - baseline_r2
            delta_rmse = result['rmse'] - baseline_rmse
            
            status = "✅" if delta_r2 > 0.005 else ("⚠️ " if abs(delta_r2) < 0.005 else "❌")
            
            print(f"{status} {feature_name:25s} → R²={result['r2']:7.4f} (Δ{delta_r2:+.4f}) | "
                  f"RMSE=${result['rmse']:5.2f} (Δ${delta_rmse:+.2f}) | "
                  f"Imp={result['importance']:.3f}")
    
    # ========== RÉCAPITULATIF ==========
    print(f"\n{'='*90}")
    print(f"📋 RÉCAPITULATIF - FEATURES PERTINENTES")
    print(f"{'='*90}\n")
    
    # Trier par amélioration R²
    results_sorted = sorted(results, key=lambda x: x['r2'], reverse=True)
    
    good_features = []
    neutral_features = []
    bad_features = []
    
    for r in results_sorted:
        delta_r2 = r['r2'] - baseline_r2
        
        if delta_r2 > 0.005:  # Amélioration significative
            good_features.append(r)
        elif delta_r2 > -0.005:  # Neutre
            neutral_features.append(r)
        else:  # Dégradation
            bad_features.append(r)
    
    print(f"✅ FEATURES À CONSERVER ({len(good_features)}):")
    if good_features:
        for r in good_features:
            delta_r2 = r['r2'] - baseline_r2
            print(f"   • {r['name']:25s} → R²={r['r2']:.4f} (Δ{delta_r2:+.4f}) | Imp={r['importance']:.3f}")
            print(f"     Colonnes: {', '.join(r['columns'])}")
    else:
        print("   Aucune feature n'améliore significativement le modèle")
    
    print(f"\n⚠️  FEATURES NEUTRES ({len(neutral_features)}):")
    if neutral_features:
        for r in neutral_features[:5]:  # Top 5
            delta_r2 = r['r2'] - baseline_r2
            print(f"   • {r['name']:25s} → R²={r['r2']:.4f} (Δ{delta_r2:+.4f})")
    else:
        print("   Aucune")
    
    print(f"\n❌ FEATURES À SUPPRIMER ({len(bad_features)}):")
    if bad_features:
        for r in bad_features:
            delta_r2 = r['r2'] - baseline_r2
            print(f"   • {r['name']:25s} → R²={r['r2']:.4f} (Δ{delta_r2:+.4f}) [DÉGRADE]")
    else:
        print("   Aucune")
    
    # ========== MODÈLE OPTIMAL ==========
    if good_features:
        print(f"\n{'='*90}")
        print(f"🚀 TEST DU MODÈLE OPTIMAL (BASE + TOUTES LES BONNES FEATURES)")
        print(f"{'='*90}\n")
        
        optimal_features = base_features.copy()
        for r in good_features:
            optimal_features.extend(r['columns'])
        
        X_optimal = df[optimal_features]
        X_train_opt = X_optimal.iloc[:split]
        X_test_opt = X_optimal[split:]
        
        model_opt = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
        model_opt.fit(X_train_opt, y_train)
        preds_opt = model_opt.predict(X_test_opt)
        
        optimal_r2 = r2_score(y_test, preds_opt)
        optimal_rmse = np.sqrt(mean_squared_error(y_test, preds_opt))
        optimal_mae = mean_absolute_error(y_test, preds_opt)
        
        print(f"📊 MODÈLE OPTIMAL ({len(optimal_features)} features):")
        print(f"   R²   = {optimal_r2:.4f} (Δ{optimal_r2 - baseline_r2:+.4f})")
        print(f"   RMSE = ${optimal_rmse:.2f} (Δ${optimal_rmse - baseline_rmse:+.2f})")
        print(f"   MAE  = ${optimal_mae:.2f} (Δ${optimal_mae - baseline_mae:+.2f})")
        print(f"\n   Features: {optimal_features}")
        
        # Sauvegarder la liste
        with open('optimal_features.txt', 'w') as f:
            f.write("# FEATURES OPTIMALES - XGBoost TotalEnergies\n")
            f.write(f"# R² = {optimal_r2:.4f} | RMSE = ${optimal_rmse:.2f}\n\n")
            for feat in optimal_features:
                f.write(f"{feat}\n")
        
        print(f"\n✅ Liste sauvegardée: optimal_features.txt")
    
    print(f"\n{'='*90}")
    print(f"✅ TEST TERMINÉ")
    print(f"{'='*90}\n")
    
    return {
        'baseline': {'r2': baseline_r2, 'rmse': baseline_rmse, 'mae': baseline_mae},
        'good_features': good_features,
        'neutral_features': neutral_features,
        'bad_features': bad_features
    }


if __name__ == "__main__":
    results = run_comprehensive_test(ticker="TTE.PA", name="TotalEnergies", prediction_days=5)
