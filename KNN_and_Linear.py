'''Baseline ML algorithms: Linear Regression & KNN Regression'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from data import load_eurusd_data


def run_baseline_models():
    """Exécute Linear Regression et KNN Regression comme baseline"""
    
    # Chargement des données depuis data.py
    df = load_eurusd_data()
    
    # Split temporel
    train_size = int(len(df) * 0.8)
    X = df[['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_5']]
    y = df['Target']
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Taille dataset: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}\n")
    
    # Linear Regression
    print("="*60)
    print("LINEAR REGRESSION")
    print("="*60)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print(f"RMSE: {rmse_lr:.6f} | MAE: {mae_lr:.6f} | R²: {r2_lr:.6f}")
    cv_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"CV RMSE: {np.sqrt(-cv_lr.mean()):.6f} (+/- {np.sqrt(cv_lr.std()):.6f})\n")
    
    # KNN Regression
    print("="*60)
    print("KNN REGRESSION")
    print("="*60)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    rmse_knn = np.sqrt(mse_knn)
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)
    
    print(f"RMSE: {rmse_knn:.6f} | MAE: {mae_knn:.6f} | R²: {r2_knn:.6f}")
    cv_knn = cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"CV RMSE: {np.sqrt(-cv_knn.mean()):.6f} (+/- {np.sqrt(cv_knn.std()):.6f})\n")
    
    # Visualisation
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_test.values[:100], label='Réel', marker='o')
    plt.plot(y_pred_lr[:100], label='Linear Regression', marker='x')
    plt.title('Linear Regression')
    plt.xlabel('Échantillons')
    plt.ylabel('Prix EURUSD')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(y_test.values[:100], label='Réel', marker='o')
    plt.plot(y_pred_knn[:100], label='KNN', marker='x')
    plt.title('KNN Regression')
    plt.xlabel('Échantillons')
    plt.ylabel('Prix EURUSD')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison.png')
    plt.show()
    
    # Résumé
    print("="*60)
    print("RÉSUMÉ COMPARATIF")
    print("="*60)
    results = pd.DataFrame({
        'Modèle': ['Linear Regression', 'KNN Regression'],
        'RMSE': [rmse_lr, rmse_knn],
        'MAE': [mae_lr, mae_knn],
        'R²': [r2_lr, r2_knn]
    })
    print(results.to_string(index=False))
    
    # Retourner les résultats pour main.py
    return {
        'results_df': results,
        'lr_predictions': y_pred_lr,
        'knn_predictions': y_pred_knn,
        'y_test': y_test,
        'lr_metrics': {'RMSE': rmse_lr, 'MAE': mae_lr, 'R²': r2_lr},
        'knn_metrics': {'RMSE': rmse_knn, 'MAE': mae_knn, 'R²': r2_knn}
    }