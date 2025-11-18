'''LSTM (Long Short-Term Memory) for time series forecasting - Deep Learning approach'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from data import load_eurusd_data
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️ TensorFlow non installé. Installez avec: pip install tensorflow")


def create_sequences(data, seq_length=10):
    """Créer des séquences pour LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def run_lstm_model():
    """Exécute LSTM pour la prédiction de séries temporelles"""
    
    if not LSTM_AVAILABLE:
        print("LSTM n'est pas disponible. Annulation.")
        return None
    
    # Chargement des données
    df = load_eurusd_data()
    
    # Utiliser la colonne Close pour LSTM
    ts_data = df['Close'].values.reshape(-1, 1)
    
    # Normalisation (crucial pour les réseaux de neurones)
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_data_scaled = scaler.fit_transform(ts_data)
    
    # Créer des séquences (utilise les 10 derniers jours pour prédire le suivant)
    seq_length = 10
    X, y = create_sequences(ts_data_scaled, seq_length)
    
    # Split temporel (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape pour LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    try:
        # Construction du modèle LSTM
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Early stopping pour éviter l'overfitting
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Entraînement silencieux
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop]
        )
        
        # Prédictions
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Dénormaliser les prédictions
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_test_original = scaler.inverse_transform(y_test).flatten()
        
        # Calcul des métriques
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        # Calcul de l'accuracy (direction correcte)
        direction_actual = np.diff(y_test_original) > 0
        direction_pred = np.diff(y_pred) > 0
        accuracy = np.mean(direction_actual == direction_pred) * 100
        
        # Visualisation
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(y_test_original[:100], label='Réel', marker='o')
            plt.plot(y_pred[:100], label='LSTM', marker='x')
            plt.title(f'LSTM - 100 premières prédictions | Accuracy: {accuracy:.2f}% | R²: {r2:.4f}')
            plt.xlabel('Échantillons')
            plt.ylabel('Prix EURUSD')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('lstm_predictions.png')
            plt.show()
        except Exception:
            pass
        
        # Retourner les résultats
        return {
            'predictions': y_pred,
            'y_test': y_test_original,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Accuracy': accuracy},
            'model': model
        }
        
    except Exception as e:
        print(f"⚠️ LSTM a rencontré une erreur: {e}")
        return None


if __name__ == "__main__":
    print("🧠 Exécution du modèle LSTM\n")
    results = run_lstm_model()
    if results:
        print("\n✅ Analyse LSTM terminée!")
