import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker="EURUSD=X", name="EUR/USD", prediction_window=5):
    """Télécharge les données et prépare des features avancées"""
    print(f"\n📥 Téléchargement des données: {name} ({ticker})")
    df = yf.download(ticker, start="2010-01-01", end="2025-01-01", progress=False, auto_adjust=True)
    
    # S'assurer que c'est un DataFrame simple (pas multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Feature engineering avancé
    # 1. Prix et variations
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Moyennes mobiles (tendances)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # 3. Indicateurs techniques
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # 4. Volume (seulement pour les actions)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Trend'] = df['Volume'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False)
    
    # 5. Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # 6. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # 7. Lags multiples
    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Close'].shift(i)
        df[f'Return_Lag_{i}'] = df['Returns'].shift(i)
    
    # 8. Features temporelles (patterns calendaires)
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Is_Monday'] = (df.index.dayofweek == 0).astype(int)
    df['Is_Friday'] = (df.index.dayofweek == 4).astype(int)
    df['Is_Month_Start'] = (df.index.day <= 5).astype(int)
    df['Is_Month_End'] = (df.index.day >= 25).astype(int)
    df['Is_January'] = (df.index.month == 1).astype(int)
    
    # 9. Patterns de chandeliers
    df['Is_Green_Day'] = (df['Close'] > df['Open']).astype(int)
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    # 10. Tendances consécutives
    df['Consecutive_Up'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df['Up_Streak'] = df['Consecutive_Up'].groupby((df['Consecutive_Up'] != df['Consecutive_Up'].shift()).cumsum()).cumsum()
    
    # Target : PREDICTION SUR FENETRE PLUS LONGUE
    df['Future_Return'] = (df['Close'].shift(-prediction_window) - df['Close']) / df['Close']
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    
    df = df.dropna()
    print(f"✅ {len(df)} jours de données chargées")
    print(f"🎯 Prédiction sur {prediction_window} jours")
    return df

# Fonction de compatibilité
def load_eurusd_data():
    return load_data("EURUSD=X", "EUR/USD")

def compute_rsi(series, period=14):
    """Calcule le RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    # Test avec EURUSD
    df = load_data("EURUSD=X", "EUR/USD")
    df.to_excel("EURUSD_Features.xlsx")
    print("Fichier Excel créé : EURUSD_Features.xlsx")
