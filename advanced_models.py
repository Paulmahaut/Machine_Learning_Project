'''Advanced ML algorithms: Random Forest, XGBoost, LightGBM'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from data import load_data
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost non installé. pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM non installé. pip install lightgbm")


def run_advanced_models(ticker="EURUSD=X", name="EUR/USD", prediction_window=5):
    """Exécute Random Forest, XGBoost et LightGBM"""
    
    # Chargement des données
    df = load_data(ticker, name, prediction_window)
    
    # Features : toutes sauf Target et Close
    feature_cols = [col for col in df.columns if col not in ['Target', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Future_Return']]
    
    # Split temporel
    train_size = int(len(df) * 0.8)
    X = df[feature_cols]
    y = df['Target']
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    results = {}
    
    print("\n" + "="*60)
    print("🚀 MODÈLES AVANCÉS - CLASSIFICATION")
    print("="*60)
    
    # ============================================
    # 1. RANDOM FOREST
    # ============================================
    print("\n🌲 Entraînement Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    rf_acc = accuracy_score(y_test, y_pred_rf) * 100
    print(f"✅ Random Forest → Accuracy: {rf_acc:.2f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 5 features importantes:")
    print(feature_importance.head(5).to_string(index=False))
    
    results['rf'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'accuracy': rf_acc,
        'feature_importance': feature_importance
    }
    
    # ============================================
    # 2. GRADIENT BOOSTING
    # ============================================
    print("\n📈 Entraînement Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    gb_acc = accuracy_score(y_test, y_pred_gb) * 100
    print(f"✅ Gradient Boosting → Accuracy: {gb_acc:.2f}%")
    
    results['gb'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'accuracy': gb_acc
    }
    
    # ============================================
    # 3. XGBOOST
    # ============================================
    if XGBOOST_AVAILABLE:
        print("\n⚡ Entraînement XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        
        xgb_acc = accuracy_score(y_test, y_pred_xgb) * 100
        print(f"✅ XGBoost → Accuracy: {xgb_acc:.2f}%")
        
        results['xgb'] = {
            'model': xgb_model,
            'predictions': y_pred_xgb,
            'accuracy': xgb_acc
        }
    
    # ============================================
    # 4. LIGHTGBM
    # ============================================
    if LIGHTGBM_AVAILABLE:
        print("\n💡 Entraînement LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        
        lgb_acc = accuracy_score(y_test, y_pred_lgb) * 100
        print(f"✅ LightGBM → Accuracy: {lgb_acc:.2f}%")
        
        results['lgb'] = {
            'model': lgb_model,
            'predictions': y_pred_lgb,
            'accuracy': lgb_acc
        }
    
    # ============================================
    # COMPARAISON
    # ============================================
    print("\n" + "="*60)
    print("📊 RÉSULTATS FINAUX")
    print("="*60)
    for name, res in results.items():
        print(f"{name.upper():12} → Accuracy: {res['accuracy']:.2f}%")
    
    # Meilleur modèle
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Meilleur modèle: {best_model[0].upper()} ({best_model[1]['accuracy']:.2f}%)")
    
    # Matrice de confusion du meilleur
    print("\n📋 Rapport de classification (meilleur modèle):")
    print(classification_report(y_test, best_model[1]['predictions']))
    
    # Visualisation
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Comparaison accuracy
    plt.subplot(1, 2, 1)
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    colors = ['#2ecc71' if acc >= 60 else '#e74c3c' for acc in accuracies]
    plt.bar(models, accuracies, color=colors)
    plt.axhline(y=60, color='blue', linestyle='--', label='Objectif 60%')
    plt.axhline(y=50, color='red', linestyle='--', label='Hasard 50%')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparaison des modèles')
    plt.legend()
    plt.ylim([0, 100])
    
    # Subplot 2: Matrice de confusion
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, best_model[1]['predictions'])
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Matrice de confusion - {best_model[0].upper()}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Baisse', 'Hausse'])
    plt.yticks(tick_marks, ['Baisse', 'Hausse'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.ylabel('Vraie classe')
    plt.xlabel('Prédiction')
    
    plt.tight_layout()
    plt.savefig('advanced_models_results.png', dpi=150, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé: advanced_models_results.png")
    plt.show()
    
    return results


if __name__ == "__main__":
    run_advanced_models()
