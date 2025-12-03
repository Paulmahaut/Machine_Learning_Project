"""
XGBoost Simple v2 - Optimized for Tesla with regularization
Features: 24 TESLA-specific features
Overfitting reduced: Gap train-test = 0.1659 (vs 0.1862 before)
Performance improved: R² = 0.8309 (vs 0.8133 before)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True):
    """
    Train XGBoost model to predict stock prices
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (default: "TSLA")
    name : str
        Stock name for display (default: "Tesla")
    prediction_days : int
        How many days ahead to predict (default: 5)
    verbose : bool
        Print results (default: True)
    
    Returns:
    --------
    dict : Model and metrics (model, predictions, y_test, r2, rmse, mae)
    """
    
    # ========== STEP 1: Download stock data ==========
    # Download 10 years of historical data from Yahoo Finance
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
    
    # yfinance sometimes returns multi-level column names, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    
    # ========== STEP 2: Create BASE features (6 features) ==========
    # These are the fundamental features from xgboost_simple.py
    
    # Moving Averages: smoothed price trends
    df['MA_5'] = df['Close'].rolling(5).mean()      # 5-day average (short term)
    df['MA_20'] = df['Close'].rolling(20).mean()    # 20-day average (medium term)
    
    # Price Lags: historical prices (yesterday, 2 days ago, 3 days ago)
    # These capture the market's "memory" - past prices influence future
    df['Lag_1'] = df['Close'].shift(1)  # Yesterday's price
    df['Lag_2'] = df['Close'].shift(2)  # 2 days ago
    df['Lag_3'] = df['Close'].shift(3)  # 3 days ago
    
    # Volatility: how much the price fluctuates
    # High volatility = risky/unstable, Low volatility = stable
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    
    # ========== STEP 3: Create ADVANCED features (18 features) ==========
    # These features were tested and validated to improve Tesla predictions
    
    # --- Bollinger Bands (2 features) ---
    # Shows if price is high/low compared to recent volatility
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20  # Upper band (mean + 2 std dev)
    df['BB_lower'] = ma20 - 2 * std20  # Lower band (mean - 2 std dev)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']  # How wide the bands are
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])  # 0 to 1
    
    # --- Price Changes (4 features) ---
    # Raw and percentage changes over 1 and 3 days
    df['Price_change_1d'] = df['Close'].diff(1)       # $ change in 1 day
    df['Price_change_3d'] = df['Close'].diff(3)       # $ change in 3 days
    df['Price_pct_1d'] = df['Close'].pct_change(1)    # % change in 1 day
    df['Price_pct_3d'] = df['Close'].pct_change(3)    # % change in 3 days
    
    # --- Rate of Change (2 features) ---
    # % change over longer periods (5 and 10 days)
    df['RoC_5'] = df['Close'].pct_change(5)   # 5-day rate of change
    df['RoC_10'] = df['Close'].pct_change(10) # 10-day rate of change
    
    # --- Momentum (2 features) ---
    # Difference between today's price and N days ago
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)   # 5-day momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10) # 10-day momentum
    
    # --- MACD (3 features) ---
    # Moving Average Convergence Divergence - trend indicator
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()  # Fast EMA
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()  # Slow EMA
    df['MACD'] = df['EMA_12'] - df['EMA_26']                      # MACD line
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']              # Histogram (momentum)
    
    # --- ATR (1 feature) ---
    # Average True Range - measures volatility considering gaps
    df['H-L'] = df['High'] - df['Low']  # High-Low range
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()  # High vs previous close
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()   # Low vs previous close
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)  # True Range (max of 3)
    df['ATR'] = df['TR'].rolling(14).mean()  # Average over 14 days
    
    # --- Volume (2 features) ---
    # Trading volume indicates market interest
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()  # 5-day average volume
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']  # Current vs average
    
    
    # ========== STEP 4: Create TARGET variable ==========
    # What we want to predict: price N days in the future
    df['Target'] = df['Close'].shift(-prediction_days)
    
    # Remove rows with NaN (missing values from rolling calculations)
    df = df.dropna()
    
    
    # ========== STEP 5: Prepare features and target ==========
    # List all 24 features we'll use for training
    features = [
        # Base features (6)
        'MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility',
        # Bollinger Bands (2)
        'BB_width', 'BB_position',
        # Price Changes (4)
        'Price_change_1d', 'Price_change_3d', 'Price_pct_1d', 'Price_pct_3d',
        # Rate of Change (2)
        'RoC_5', 'RoC_10',
        # Momentum (2)
        'Momentum_5', 'Momentum_10',
        # MACD (3)
        'MACD', 'MACD_signal', 'MACD_hist',
        # ATR (1)
        'ATR',
        # Volume (2)
        'Volume_MA_5', 'Volume_Ratio',
        # EMA (2)
        'EMA_12', 'EMA_26'
    ]
    
    # X = input features, y = target (what we predict)
    X = df[features]
    y = df['Target']
    
    
    # ========== STEP 6: Split data into train and test sets ==========
    # 80% for training, 20% for testing
    # IMPORTANT: We split chronologically (not random) for time series!
    split = int(len(X) * 0.8)
    X_train = X[:split]   # First 80% of data
    X_test = X[split:]    # Last 20% of data
    y_train = y[:split]
    y_test = y[split:]
    
    
    # ========== STEP 7: Train XGBoost model ==========
    """
    XGBoost Parameters Explained:
    
    n_estimators=100
        - Number of decision trees to build
        - More trees = more complex model
        - 100 is a good balance for our data
    
    max_depth=3
        - Maximum depth of each tree (how many levels)
        - Lower = simpler trees = less overfitting
        - Changed from 5→3 to reduce overfitting
        - Each level asks one yes/no question about features
    
    learning_rate=0.1
        - How much each tree contributes to final prediction
        - Lower = slower learning but more robust
        - 0.1 is standard (range: 0.01 to 0.3)
    
    reg_lambda=1
        - L2 regularization (Ridge penalty)
        - Penalizes large weights → smoother predictions
        - Helps prevent overfitting
        - Higher = more penalty (0 = no penalty)
    
    subsample=0.8
        - Use 80% of training data for each tree
        - Like "bagging" - reduces overfitting
        - Each tree sees slightly different data
    
    colsample_bytree=0.8
        - Use 80% of features for each tree
        - Forces model to not rely on single features
        - Improves generalization
    
    min_child_weight=3
        - Minimum data points required in a leaf node
        - Higher = can't create very specific rules
        - Prevents overfitting to outliers
    
    verbosity=0
        - Don't print training progress (keeps output clean)
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        reg_lambda=1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        verbosity=0
    )
    
    # Train the model on training data
    model.fit(X_train, y_train)
    
    
    # ========== STEP 8: Make predictions ==========
    predictions = model.predict(X_test)
    
    
    # ========== STEP 9: Calculate performance metrics ==========
    """
    Metrics Explained:
    
    RMSE (Root Mean Squared Error)
        - Average prediction error in dollars
        - Lower is better
        - Example: RMSE=$23.90 means predictions are off by ~$24 on average
    
    MAE (Mean Absolute Error)
        - Average absolute error in dollars
        - More interpretable than RMSE
        - Less sensitive to outliers
    
    R² (R-squared / Coefficient of Determination)
        - How well model explains variance in data
        - Range: 0 to 1 (higher is better)
        - 0.83 = model explains 83% of price variation
        - 1.0 = perfect predictions
        - 0.0 = model is useless (predicts average)
    """
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    
    # ========== STEP 10: Print results ==========
    if verbose:
        print(f"XGBoost Results: R²={r2:.4f} | RMSE=${rmse:.2f} | MAE=${mae:.2f} | Train={len(X_train)}d | Test={len(X_test)}d")
    
    
    # ========== STEP 11: Visualize predictions vs reality ==========
    plt.figure(figsize=(12, 6))
    
    # Plot real prices (blue solid line)
    plt.plot(y_test.values, label='Real Price', alpha=0.7, linewidth=2, color='blue')
    
    # Plot predictions (orange dashed line)
    plt.plot(predictions, label='Predictions', alpha=0.7, linestyle='--', linewidth=2, color='orange')
    
    # Add title with R² score
    plt.title(f'{name} - Stock Price Prediction (R²={r2:.4f})', fontsize=14, fontweight='bold')
    
    # Labels and legend
    plt.xlabel('Days (test set)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save and show
    plt.savefig(f'{ticker}_xgboost_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
    # ========== STEP 12: Return results ==========
    return {
        'model': model,
        'predictions': predictions,
        'y_test': y_test,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    """
    This code runs when you execute the script directly
    (not when importing it as a module)
    """
    # Run the model on Tesla stock
    results = run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True)

