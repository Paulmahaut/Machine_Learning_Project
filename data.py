import yfinance as yf
import pandas as pd

def load_eurusd_data():
    """Télécharge les données EURUSD et prépare les features"""
    df = yf.download("EURUSD=X", start="2005-01-01", end="2025-01-01", progress=False)
    
    # Feature engineering
    df['Target'] = df['Close'].shift(-1)  # Prix du lendemain
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Rolling_Mean_5'] = df['Close'].rolling(window=5).mean()
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = load_eurusd_data()
    df.to_excel("EURUSD_2005_2025.xlsx")
    print("Fichier Excel créé : EURUSD_2005_2025.xlsx")
