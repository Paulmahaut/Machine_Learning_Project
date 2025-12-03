# 📚 Antisèche - Toutes les Features XGBoost Simple v2

## 🔵 **BASE FEATURES (6)** - Fondamentales

### 1. **MA_5 & MA_20** (Moving Averages - Moyennes Mobiles)
```python
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
```
- **Quoi ?** Moyenne du prix sur les N derniers jours
- **Pourquoi ?** Lisse les fluctuations, montre la tendance
- **Signal :** 
  - Prix > MA → tendance haussière
  - Prix < MA → tendance baissière
- **Différence :** MA_5 (court terme) vs MA_20 (moyen terme)
- **Importance :** MA_5 = 20.75% | MA_20 = 0.37%

### 2. **Lag_1, Lag_2, Lag_3** (Prix historiques)
```python
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
```
- **Quoi ?** Prix d'hier, avant-hier, il y a 3 jours
- **Pourquoi ?** Les marchés ont une mémoire = autocorrélation
- **Usage :** Lag_1 est LA feature la plus importante (48.78%)
- **Importance :** Lag_1 = 48.78% | Lag_2 = 0.09% | Lag_3 = 0.07%

### 3. **Volatility** (Volatilité)
```python
df['Volatility'] = df['Close'].pct_change().rolling(20).std()
```
- **Quoi ?** Écart-type des variations de prix sur 20 jours
- **Pourquoi ?** Mesure l'instabilité/risque
- **Signal :** 
  - Volatilité haute → marché instable
  - Basse → stable
- **Importance :** 0.05%

---

## 🟢 **BOLLINGER BANDS (2)** - Volatilité dynamique

### 4. **BB_width & BB_position**
```python
ma20 = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
BB_upper = ma20 + 2 * std20
BB_lower = ma20 - 2 * std20
df['BB_width'] = BB_upper - BB_lower
df['BB_position'] = (df['Close'] - BB_lower) / (BB_upper - BB_lower)
```
- **Quoi ?** Bandes à ±2 écarts-types autour de MA_20
- **BB_width :** Largeur des bandes (mesure de volatilité)
- **BB_position :** Position du prix entre les bandes (0 à 1)
- **Signaux classiques :**
  - BB squeeze (bandes serrées) → explosion imminente
  - Prix touche bande haute → surachat potentiel
  - Prix touche bande basse → survente potentielle
- **Importance :** BB_width = 0.09% | BB_position = 0.15%

---

## 🟡 **PRICE CHANGES (4)** - Variations brutes

### 5. **Price_change_1d & Price_change_3d**
```python
df['Price_change_1d'] = df['Close'].diff(1)
df['Price_change_3d'] = df['Close'].diff(3)
```
- **Quoi ?** Différence absolue de prix (en $)
- **Pourquoi ?** Capture l'amplitude du mouvement
- **Importance :** 1d = 0.03% | 3d = 0.06%

### 6. **Price_pct_1d & Price_pct_3d**
```python
df['Price_pct_1d'] = df['Close'].pct_change(1)
df['Price_pct_3d'] = df['Close'].pct_change(3)
```
- **Quoi ?** Variation en % (normalisée)
- **Pourquoi ?** Comparable entre différents actifs
- **Exemple :** +3% = 0.03
- **Importance :** 1d = 0.04% | 3d = 0.16%

---

## 🟠 **RATE OF CHANGE (2)** - Momentum pur

### 7. **RoC_5 & RoC_10**
```python
df['RoC_5'] = df['Close'].pct_change(5)
df['RoC_10'] = df['Close'].pct_change(10)
```
- **Quoi ?** Variation % sur 5 ou 10 jours
- **Pourquoi ?** Mesure la vitesse du mouvement
- **Différence avec Price_pct :** Sur période plus longue (5/10j vs 1/3j)
- **Importance :** RoC_5 = 0.17% | RoC_10 = 0.06%

---

## 🔴 **MOMENTUM (2)** - Force du mouvement

### 8. **Momentum_5 & Momentum_10**
```python
df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
```
- **Quoi ?** Différence de prix entre aujourd'hui et il y a N jours
- **Pourquoi ?** Identique à Price_change mais sur 5/10j
- **Signal :** Momentum > 0 → mouvement haussier qui continue
- **Importance :** Momentum_5 = 0.17% | Momentum_10 = 0.02%

---

## 🟣 **MACD (3)** - Convergence/Divergence des moyennes

### 9. **EMA_12 & EMA_26**
```python
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
```
- **Quoi ?** Moyenne Mobile Exponentielle (EMA)
- **Différence avec MA :** Donne plus de poids aux prix récents
- **Pourquoi 12 & 26 ?** Standard en finance (court/moyen terme)
- **Importance :** EMA_12 = 0.32% | EMA_26 = 27.97% ⭐

### 10. **MACD, MACD_signal, MACD_hist**
```python
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']
```
- **MACD :** Divergence des moyennes (momentum)
- **MACD_signal :** Moyenne du MACD (ligne de signal)
- **MACD_hist :** Force du momentum
- **Signaux classiques :**
  - MACD > signal → achat
  - MACD < signal → vente
  - Histogram croissant → accélération haussière
- **Importance :** MACD = 0.30% | MACD_signal = 0.07% | MACD_hist = 0.12%

---

## 🟤 **ATR (1)** - Volatilité réelle

### 11. **ATR** (Average True Range)
```python
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
df['ATR'] = df['TR'].rolling(14).mean()
```
- **Quoi ?** Amplitude moyenne réelle sur 14 jours
- **Pourquoi ?** Meilleure mesure de volatilité que simple écart-type
- **Usage :** Stop-loss dynamique, détection de cassures
- **Importance :** 0.08%

---

## ⚫ **VOLUME (2)** - Intensité du marché

### 12. **Volume_MA_5 & Volume_Ratio**
```python
df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
```
- **Volume_MA_5 :** Volume moyen sur 5 jours
- **Volume_Ratio :** Volume actuel vs moyenne
- **Signal :**
  - Ratio > 1.5 → volume exceptionnel (alerte)
  - Volume élevé + hausse prix → forte conviction
  - Volume faible → mouvement suspect
- **Importance :** Volume_MA_5 = 0.07% | Volume_Ratio = 0.03%

---

## 🎯 **RÉCAPITULATIF PAR IMPORTANCE**

| Rang | Feature | Importance | Pourcentage | Catégorie |
|------|---------|------------|-------------|-----------|
| 🥇 1 | **Lag_1** | 0.4878 | **48.78%** | Base |
| 🥈 2 | **EMA_26** | 0.2797 | **27.97%** | MACD |
| 🥉 3 | **MA_5** | 0.2075 | **20.75%** | Base |
| 4 | MA_20 | 0.0037 | 0.37% | Base |
| 5 | EMA_12 | 0.0032 | 0.32% | MACD |
| 6 | MACD | 0.0030 | 0.30% | MACD |
| 7 | Momentum_5 | 0.0017 | 0.17% | Momentum |
| 8 | RoC_5 | 0.0017 | 0.17% | RoC |
| 9 | Price_pct_3d | 0.0016 | 0.16% | Price Changes |
| 10 | BB_position | 0.0015 | 0.15% | Bollinger |
| 11 | MACD_hist | 0.0012 | 0.12% | MACD |
| 12 | Lag_2 | 0.0009 | 0.09% | Base |
| 13 | BB_width | 0.0009 | 0.09% | Bollinger |
| 14 | ATR | 0.0008 | 0.08% | ATR |
| 15 | Volume_MA_5 | 0.0007 | 0.07% | Volume |
| 16 | Lag_3 | 0.0007 | 0.07% | Base |
| 17 | MACD_signal | 0.0007 | 0.07% | MACD |
| 18 | RoC_10 | 0.0006 | 0.06% | RoC |
| 19 | Price_change_3d | 0.0006 | 0.06% | Price Changes |
| 20 | Volatility | 0.0005 | 0.05% | Base |
| 21 | Price_pct_1d | 0.0004 | 0.04% | Price Changes |
| 22 | Price_change_1d | 0.0003 | 0.03% | Price Changes |
| 23 | Volume_Ratio | 0.0003 | 0.03% | Volume |
| 24 | Momentum_10 | 0.0002 | 0.02% | Momentum |

**TOTAL = 100%**

---

## 📊 **ANALYSE GROUPÉE**

| Groupe | Features | Importance Totale |
|--------|----------|-------------------|
| **TOP 3** | Lag_1, EMA_26, MA_5 | **97.50%** ⭐⭐⭐ |
| **TOP 10** | + 7 features | **99.12%** |
| **14 dernières** | Features < 0.1% | **0.88%** (quasi inutiles) |

---

## 💡 **COMMENT LIRE CES FEATURES ENSEMBLE ?**

### Exemple concret : Tesla à $250

**Données :**
- **Lag_1 = $248** → Prix monte (+$2)
- **MA_5 = $245** → Au-dessus de la moyenne court terme (haussier)
- **EMA_26 = $240** → Tendance moyen terme haussière
- **BB_position = 0.8** → Près de la bande haute (attention surachat)
- **MACD_hist > 0** → Momentum haussier
- **Volume_Ratio = 1.8** → Volume inhabituel (confirmation)
- **ATR = $15** → Volatilité élevée (risque)

**→ Prédiction probable : Continuation haussière à court terme, mais surveiller le surachat**

---

## 🔑 **POURQUOI 24 FEATURES SI SEULEMENT 3 DOMINENT ?**

**Réponse :** Les 21 autres features servent de **"tie-breakers"** quand les 3 principales sont ambiguës.

**Exemple :**
- Lag_1, EMA_26, MA_5 → signal neutre (±0%)
- → Le modèle regarde MACD, BB_position, Volume_Ratio pour trancher

**Mais** : 80-90% du temps, seules Lag_1/EMA_26/MA_5 comptent vraiment !

---

## 🎓 **POUR L'ÉVALUATION 2**

### Questions probables du prof :

**Q: "Pourquoi 24 features ?"**
→ *"Chaque feature a été testée individuellement. On a gardé uniquement celles qui améliorent le R² sur Tesla. Cependant, l'analyse d'importance montre que 3 features dominent (97.5%), les autres servent de raffinement."*

**Q: "Quelle est la feature la plus importante ?"**
→ *"Lag_1 (prix d'hier) avec 48.78% d'importance. Cela prouve que les marchés ont une forte autocorrélation à court terme."*

**Q: "Pourquoi certaines features ont une importance très faible ?"**
→ *"XGBoost se concentre sur les features les plus discriminantes. Les features à faible importance servent de 'tie-breakers' dans les cas ambigus, mais ne sont pas utilisées majoritairement."*

**Q: "Comment choisir les bonnes features ?"**
→ *"Test individuel + validation croisée + analyse d'importance post-entraînement. L'optimisation doit être asset-specific (TotalEnergies ≠ Tesla)."*

---

## 📝 **RÉSUMÉ EN 3 POINTS**

1. **24 features testées et validées** (chacune améliore le R² individuellement)
2. **3 features dominent** (Lag_1 48.78%, EMA_26 27.97%, MA_5 20.75%) = 97.5% importance
3. **Asset-specific** : Les features optimales pour Tesla ≠ TotalEnergies

**Résultat final : R² = 0.8133 sur Tesla (+5.5% vs baseline à 6 features)**
