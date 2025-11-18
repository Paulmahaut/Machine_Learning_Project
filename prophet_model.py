'''Prophet algorithm for time series forecasting - Non-taught algorithm exploration'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import load_eurusd_data

try:
    from prophet import Prophet
    import matplotlib.pyplot as plt
    import logging
    # Désactiver les logs verbeux de Prophet
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
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
    
    # Entraînement du modèle Prophet (configuration simplifiée)
    try:
        # Configuration minimaliste pour éviter les erreurs
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,  # Désactivé pour plus de stabilité
            changepoint_prior_scale=0.05,  # Réduire la sensibilité
            interval_width=0.95
        )
        
        # Entraînement silencieux
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train)
            
    except Exception as e:
        print(f"⚠️ Prophet a rencontré une erreur: {str(e)[:100]}")
        print("Prophet ne peut pas être exécuté sur cet environnement.")
        return None
    
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
    
    # Calcul de l'accuracy (direction correcte)
    direction_actual = np.diff(y_test) > 0
    direction_pred = np.diff(y_pred) > 0
    accuracy = np.mean(direction_actual == direction_pred) * 100
    
    # Visualisation simple
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(y_test[:100], label='Réel', marker='o')
        plt.plot(y_pred[:100], label='Prophet', marker='x')
        plt.title(f'Prophet - 100 premières prédictions | Accuracy: {accuracy:.2f}% | R²: {r2:.4f}')
        plt.xlabel('Échantillons')
        plt.ylabel('Prix EURUSD')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('prophet_predictions.png')
        plt.show()
    except Exception:
        pass
    
    # Sauvegarder et préparer le dataframe de forecast minimal
    try:
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.to_csv('prophet_forecast.csv', index=False)
    except Exception:
        forecast_df = forecast[['ds', 'yhat']].copy()

    # Retourner les résultats
    return {
        'predictions': y_pred,
        'y_test': y_test,
        'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Accuracy': accuracy},
        'forecast_df': forecast_df
    }


if __name__ == "__main__":
    print("🔮 Exécution du modèle Prophet\n")
    results = run_prophet_model()
    if results:
        print("\n✅ Analyse Prophet terminée!")
