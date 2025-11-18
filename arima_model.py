'''ARIMA algorithm for time series forecasting - Non-taught algorithm exploration'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import load_eurusd_data
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
    import matplotlib.pyplot as plt
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ statsmodels/pmdarima non installé. Installez avec: pip install statsmodels pmdarima")


def run_arima_model():
    """Exécute ARIMA pour la prédiction de séries temporelles"""
    
    if not ARIMA_AVAILABLE:
        print("ARIMA n'est pas disponible. Annulation.")
        return None
    
    # Chargement des données
    df = load_eurusd_data()
    
    # Utiliser la colonne Close pour ARIMA avec index temporel
    ts_data = df['Close'].copy()
    # Réinitialiser l'index pour éviter les warnings
    ts_data = ts_data.reset_index(drop=True)
    
    # Split temporel (80/20)
    train_size = int(len(ts_data) * 0.8)
    train = ts_data[:train_size]
    test = ts_data[train_size:]
    
    # Entraînement du modèle ARIMA avec auto_arima pour trouver les meilleurs paramètres
    
    try:
        # Auto ARIMA pour trouver les meilleurs paramètres automatiquement
        print("Recherche des meilleurs paramètres ARIMA (peut prendre quelques secondes)...")
        auto_model = auto_arima(
            train, 
            start_p=1, start_q=1,
            max_p=5, max_q=5, max_d=2,
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        print(f"Meilleurs paramètres trouvés: ARIMA{auto_model.order}")
        
        # Utiliser les meilleurs paramètres
        model = ARIMA(train, order=auto_model.order)
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
        
        # Calcul de l'accuracy (direction correcte)
        # Pour la prédiction de séries temporelles, on mesure si on prédit correctement la direction
        direction_actual = np.diff(y_test) > 0
        direction_pred = np.diff(y_pred) > 0
        accuracy = np.mean(direction_actual == direction_pred) * 100
        
        # Visualisation simple
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(y_test[:100], label='Réel', marker='o')
            plt.plot(y_pred[:100], label='ARIMA', marker='x')
            plt.title(f'ARIMA - 100 premières prédictions | Accuracy: {accuracy:.2f}% | R²: {r2:.4f}')
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
            'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Accuracy': accuracy},
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
