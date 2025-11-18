'''Prophet algorithm for time series forecasting - Non-taught algorithm exploration'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import load_eurusd_data

try:
    from prophet import Prophet
    import matplotlib.pyplot as plt
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet non installé. Installez avec: pip install prophet")


def run_prophet_model():
    """Exécute Prophet pour la prédiction de séries temporelles"""
    
    if not PROPHET_AVAILABLE:
        print("Prophet n'est pas disponible. Annulation.")
        return None
    
    # Chargement des données
    df = load_eurusd_data()
    
    # Préparer les données pour Prophet (colonnes 'ds' et 'y' requises)
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    
    # Split temporel (80/20)
    train_size = int(len(df_prophet) * 0.8)
    train = df_prophet[:train_size]
    test = df_prophet[train_size:]
    
    print(f"Taille dataset: {len(df_prophet)} | Train: {len(train)} | Test: {len(test)}\n")
    
    # Entraînement du modèle Prophet (simplifié)
    print("="*60)
    print("PROPHET TIME SERIES FORECASTING")
    print("="*60)
    
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    model.fit(train)
    
    # Prédictions
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    
    # Extraire les prédictions pour le test set
    y_pred = forecast['yhat'][-len(test):].values
    y_test = test['y'].values
    
    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}\n")
    
    # Visualisation simple
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:100], label='Réel', marker='o')
    plt.plot(y_pred[:100], label='Prophet', marker='x')
    plt.title('Prophet - 100 premières prédictions')
    plt.xlabel('Échantillons')
    plt.ylabel('Prix EURUSD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prophet_predictions.png')
    plt.show()
    
    # Retourner les résultats
    return {
        'predictions': y_pred,
        'y_test': y_test,
        'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    }


if __name__ == "__main__":
    print("🔮 Exécution du modèle Prophet\n")
    results = run_prophet_model()
    if results:
        print("\n✅ Analyse Prophet terminée!")
