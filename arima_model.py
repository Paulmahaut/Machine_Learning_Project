'''ARIMA algorithm for time series forecasting - Non-taught algorithm exploration'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import load_eurusd_data
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ statsmodels non installé. Installez avec: pip install statsmodels")


def run_arima_model():
    """Exécute ARIMA pour la prédiction de séries temporelles"""
    
    if not ARIMA_AVAILABLE:
        print("ARIMA n'est pas disponible. Annulation.")
        return None
    
    # Chargement des données
    df = load_eurusd_data()
    
    # Utiliser la colonne Close pour ARIMA
    ts_data = df['Close']
    
    # Split temporel (80/20)
    train_size = int(len(ts_data) * 0.8)
    train = ts_data[:train_size]
    test = ts_data[train_size:]
    
    print(f"Taille dataset: {len(ts_data)} | Train: {len(train)} | Test: {len(test)}\n")
    
    # Entraînement du modèle ARIMA
    print("="*60)
    print("ARIMA TIME SERIES FORECASTING")
    print("="*60)
    
    try:
        # ARIMA(5,1,0) - ordre simple pour démarrer
        model = ARIMA(train, order=(5, 1, 0))
        print("Entraînement du modèle ARIMA en cours...")
        model_fit = model.fit()
        
        # Prédictions
        y_pred = model_fit.forecast(steps=len(test))
        y_test = test.values
        
        # Convertir en numpy array 1D si besoin
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        y_pred = np.asarray(y_pred).flatten()
        y_test = np.asarray(y_test).flatten()
        
        # Calcul des métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}\n")
        
        # Visualisation simple
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(y_test[:100], label='Réel', marker='o')
            plt.plot(y_pred[:100], label='ARIMA', marker='x')
            plt.title('ARIMA - 100 premières prédictions')
            plt.xlabel('Échantillons')
            plt.ylabel('Prix EURUSD')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('arima_predictions.png')
            plt.show()
        except Exception:
            pass
        
        # Créer forecast dataframe
        try:
            forecast_df = pd.DataFrame({
                'index': list(range(len(y_pred))),
                'prediction': list(y_pred),
                'actual': list(y_test)
            })
            forecast_df.to_csv('arima_forecast.csv', index=False)
        except Exception as e:
            print(f"Impossible de créer le forecast dataframe: {e}")
            forecast_df = None
        
        # Retourner les résultats
        return {
            'predictions': y_pred,
            'y_test': y_test,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2},
            'forecast_df': forecast_df
        }
        
    except Exception as e:
        print(f"⚠️ ARIMA a rencontré une erreur: {e}")
        return None


if __name__ == "__main__":
    print("📈 Exécution du modèle ARIMA\n")
    results = run_arima_model()
    if results:
        print("\n✅ Analyse ARIMA terminée!")
