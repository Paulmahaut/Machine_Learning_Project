# xgboost_simple2.py — Guide d'apprentissage ligne par ligne

Ce document explique chaque portion de code du fichier `xgboost_simple2.py`, avec contexte, objectifs, et implications. Il sert de version pédagogique exhaustive, distincte du code d'exécution.

---

## En-tête et imports

```python
"""
XGBoost Simple v2 - Optimized for Tesla with regularization
Features: 24 TESLA-specific features
Overfitting reduced: Gap train-test = 0.1659 (vs 0.1862 before)
Performance improved: R² = 0.8309 (vs 0.8133 before)
"""
```
- Bloc de documentation: décrit la version, le nombre de features, et les métriques clés.

```python
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```
- `yfinance`: téléchargement des données financières.
- `pandas`: manipulation de DataFrame.
- `numpy`: calculs numériques.
- `XGBRegressor`: modèle XGBoost pour la régression.
- `sklearn.metrics`: calcul des métriques d'évaluation.
- `matplotlib`: visualisation.
- `warnings.filterwarnings('ignore')`: évite l'affichage de messages non critiques.

---

## Fonction principale `run_xgboost`

```python
def run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True):
    """
    Train XGBoost model to predict stock prices
    ...
    """
```
- Déclare la fonction d'entraînement.
- Paramètres:
  - `ticker`: symbole boursier (par défaut Tesla).
  - `name`: nom affiché.
  - `prediction_days`: horizon de prédiction (5 jours par défaut).
  - `verbose`: contrôle l'affichage des résultats.
- Retour: dictionnaire avec le modèle et les métriques.

### STEP 1 — Téléchargement des données

```python
df = yf.download(ticker, start="2015-01-01", end="2025-01-01", progress=False)
```
- Télécharge les prix historiques entre 2015 et 2025 inclus.

```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
```
- `yfinance` peut renvoyer des colonnes multi-index (ex: `Close` sous plusieurs niveaux). On aplatit pour faciliter les accès.

### STEP 2 — Features de base (6)

```python
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
```
- Moyennes mobiles (MA) 5 et 20 jours: tendances court/moyen terme.

```python
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
```
- Décalages (lags) des prix: mémoire du marché. `Lag_1` est souvent très informatif.

```python
df['Volatility'] = df['Close'].pct_change().rolling(20).std()
```
- Volatilité: écart-type des rendements journaliers sur 20 jours.

### STEP 3 — Features avancées (18)

```python
ma20 = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BB_upper'] = ma20 + 2 * std20
df['BB_lower'] = ma20 - 2 * std20
df['BB_width'] = df['BB_upper'] - df['BB_lower']
df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
```
- Bandes de Bollinger: bornes ±2 écarts-types; `BB_width` (largeur) et `BB_position` (normalisation 0-1).

```python
df['Price_change_1d'] = df['Close'].diff(1)
df['Price_change_3d'] = df['Close'].diff(3)
df['Price_pct_1d'] = df['Close'].pct_change(1)
df['Price_pct_3d'] = df['Close'].pct_change(3)
```
- Variations absolues et relatives sur 1 et 3 jours.

```python
df['RoC_5'] = df['Close'].pct_change(5)
df['RoC_10'] = df['Close'].pct_change(10)
```
- Taux de variation sur 5 et 10 jours.

```python
df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
```
- Momentum: différence entre aujourd'hui et il y a N jours.

```python
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']
```
- MACD: indicateur de tendance basé sur EMAs; `signal` lisse le MACD; `hist` mesure le momentum.

```python
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
df['ATR'] = df['TR'].rolling(14).mean()
```
- ATR: Average True Range, mesure de volatilité prenant en compte les gaps.

```python
df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
```
- Indicateurs de volume: intérêt du marché relatif.

### STEP 4 — Variable cible

```python
df['Target'] = df['Close'].shift(-prediction_days)
```
- Décale le prix de `prediction_days` vers le futur; ici 5 jours.

```python
df = df.dropna()
```
- Supprime les lignes avec NaN (effet des `rolling`/`shift`).

### STEP 5 — Sélection des features et cible

```python
features = [
    'MA_5', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Volatility',
    'BB_width', 'BB_position',
    'Price_change_1d', 'Price_change_3d', 'Price_pct_1d', 'Price_pct_3d',
    'RoC_5', 'RoC_10',
    'Momentum_5', 'Momentum_10',
    'MACD', 'MACD_signal', 'MACD_hist',
    'ATR',
    'Volume_MA_5', 'Volume_Ratio',
    'EMA_12', 'EMA_26'
]
X = df[features]
y = df['Target']
```
- Liste explicite de 24 features utilisées.
- `X`: matrice des features; `y`: cible.

### STEP 6 — Découpage temporel train/test

```python
split = int(len(X) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]
```
- Split chronologique (80/20). Important pour séries temporelles.

### STEP 7 — Modèle XGBoost et entraînement

```python
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
model.fit(X_train, y_train)
```
- Paramètres clés et justification:
  - `n_estimators=100`: nombre d'arbres (compromis complexité/temps).
  - `max_depth=3`: limite profondeur pour réduire overfitting.
  - `learning_rate=0.1`: apprentissage progressif.
  - `reg_lambda=1`: régularisation L2.
  - `subsample=0.8`: échantillonnage des lignes (bagging).
  - `colsample_bytree=0.8`: échantillonnage des colonnes.
  - `min_child_weight=3`: taille minimale des feuilles.
  - `verbosity=0`: silence pendant l'entraînement.

### STEP 8 — Prédictions

```python
predictions = model.predict(X_test)
```
- Produit les prix prédits à `prediction_days` jours dans le futur.

### STEP 9 — Métriques

```python
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
```
- `RMSE`: erreur quadratique moyenne (en $).
- `MAE`: erreur absolue moyenne (en $).
- `R²`: part de variance expliquée.

### STEP 10 — Affichage

```python
if verbose:
    print(f"XGBoost Results: R²={r2:.4f} | RMSE=${rmse:.2f} | MAE=${mae:.2f} | Train={len(X_train)}d | Test={len(X_test)}d")
```
- Affiche les métriques et tailles des ensembles.

### STEP 11 — Visualisation

```python
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Real Price', alpha=0.7, linewidth=2, color='blue')
plt.plot(predictions, label='Predictions', alpha=0.7, linestyle='--', linewidth=2, color='orange')
plt.title(f'{name} - Stock Price Prediction (R²={r2:.4f})', fontsize=14, fontweight='bold')
plt.xlabel('Days (test set)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ticker}_xgboost_v2.png', dpi=150, bbox_inches='tight')
plt.show()
```
- Trace prix réels vs prédits; sauvegarde l'image PNG.

### STEP 12 — Retour

```python
return {
    'model': model,
    'predictions': predictions,
    'y_test': y_test,
    'r2': r2,
    'rmse': rmse,
    'mae': mae
}
```
- Retourne toutes les informations utiles.

### Bloc `__main__`

```python
if __name__ == "__main__":
    results = run_xgboost(ticker="TSLA", name="Tesla", prediction_days=5, verbose=True)
```
- Exécution directe: lance l'entraînement et affiche les résultats.

---

## Annexes — Notes pédagogiques

- Split chronologique vs aléatoire: évite le data leakage.
- Pourquoi 24 features? Toutes testées; XGBoost en retient surtout 3 (Lag_1, EMA_26, MA_5) mais les autres apportent robustesse.
- Réduction d'overfitting: profondeur, régularisation L2, échantillonnage lignes/colonnes, `min_child_weight`.
- Interprétation des métriques: RMSE/MAE en $ (intuitif), R² (~0.83) solide.
- Visualisation: essentielle pour expliquer les écarts et la dynamique.
