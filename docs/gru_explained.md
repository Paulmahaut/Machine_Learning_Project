# GRU Model — Guide pédagogique complet

## 📖 Introduction

### Qu'est-ce qu'un GRU?

**GRU = Gated Recurrent Unit** (Unité Récurrente à Portes)

C'est un type de réseau de neurones spécialisé pour les **données séquentielles** comme:
- Séries temporelles (prix d'actions, météo, etc.)
- Texte (traduction, génération)
- Audio (reconnaissance vocale)

### Différence clé vs XGBoost

| Aspect | XGBoost | GRU |
|--------|---------|-----|
| Type | Arbres de décision | Réseau de neurones récurrent |
| Input | 1 jour = 1 prédiction | Séquence de N jours → 1 prédiction |
| Mémoire | Aucune (chaque jour indépendant) | "Se souvient" des jours précédents |
| Complexité | Moyenne | Élevée |

### Exemple concret

**XGBoost:**
```
Jour 100 (features du jour 100) → Prédit jour 105
Jour 101 (features du jour 101) → Prédit jour 106
```

**GRU:**
```
Jours 70-99 (séquence de 30) → Prédit jour 105
Jours 71-100 (séquence de 30) → Prédit jour 106
```

Le GRU "lit" 30 jours d'affilée et capture les **patterns temporels** (tendances, cycles).

---

## 🔧 Préparation des données (gru_prep.py)

### Étape 1: Normalisation (CRUCIAL!)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train[features])
```

**Pourquoi normaliser?**
- Les features ont des échelles très différentes:
  - Prix: $200-$400
  - Volume: 10M-200M
  - Pourcentages: -0.1 à +0.1
- Les réseaux de neurones apprennent mieux avec des valeurs similaires
- StandardScaler transforme tout pour avoir **moyenne=0** et **écart-type=1**

**Formule:**
```
x_normalized = (x - moyenne) / écart-type
```

**Exemple:**
```
Prix brut: $250
Moyenne des prix: $300
Écart-type: $50
Prix normalisé: ($250 - $300) / $50 = -1.0
```

### Étape 2: Création des séquences

**Concept de fenêtre glissante (sliding window):**

Données brutes (après normalisation):
```
Jour 1: [feature1, feature2, ..., feature24]
Jour 2: [feature1, feature2, ..., feature24]
...
Jour 30: [feature1, feature2, ..., feature24]
Jour 31: [feature1, feature2, ..., feature24]
```

Séquences créées (SEQUENCE_LENGTH = 30):
```
Séquence 1: Jours 1-30   (30 jours × 24 features) → Target du jour 30
Séquence 2: Jours 2-31   (30 jours × 24 features) → Target du jour 31
Séquence 3: Jours 3-32   (30 jours × 24 features) → Target du jour 32
...
```

**Shape des tenseurs:**
- Input: `(n_sequences, 30, 24)`
  - `n_sequences`: nombre de séquences
  - `30`: longueur de chaque séquence (jours)
  - `24`: nombre de features par jour
- Output: `(n_sequences,)` (un prix futur par séquence)

### Étape 3: Continuité temporelle pour le test

**PIÈGE à éviter:**
Si on crée les séquences test indépendamment, la première séquence test manquerait de contexte.

**Solution:**
Concaténer les 29 derniers jours du train avec le test:
```
Train: [..., jour 1990, jour 1991, jour 1992]
                        └───────────────┘
                             ↓ Copie
Test concat: [jour 1964-1992, jour 1993, jour 1994, ...]
             └──── 29 jours ──┘
```

Ainsi, la première séquence test = jours 1964-1993 (complète!).

---

## 🏗️ Architecture du modèle (gru_train.py)

### Couche 1: GRU(64, return_sequences=True)

```python
GRU(64, return_sequences=True, input_shape=(30, 24))
```

**Que fait cette couche?**
1. Lit les 30 jours de la séquence **un par un**
2. Pour chaque jour, met à jour un "état caché" (mémoire) de taille 64
3. `return_sequences=True` → retourne l'état à chaque pas de temps

**Analogie:**
Imagine un lecteur qui lit un livre de 30 pages:
- Page 1 → Se souvient des infos importantes (64 notes mentales)
- Page 2 → Met à jour ses notes (64 valeurs)
- ...
- Page 30 → A 30 ensembles de notes (un par page)

**Output shape:** `(batch, 30, 64)`
- `30` = un état pour chaque jour
- `64` = taille de la mémoire

### Dropout(0.2)

```python
Dropout(0.2)
```

**Régularisation:**
- Désactive aléatoirement 20% des neurones pendant l'entraînement
- Force le réseau à ne pas dépendre de neurones spécifiques
- Réduit l'overfitting (comme `subsample` dans XGBoost)

### Couche 2: GRU(32, return_sequences=False)

```python
GRU(32, return_sequences=False)
```

**Que fait cette couche?**
1. Prend les 30 états de la couche précédente
2. Les condense en UN seul état final de taille 32
3. `return_sequences=False` → retourne seulement le dernier état

**Analogie:**
Le lecteur relit ses 30 pages de notes et écrit un **résumé final** (32 points clés).

**Output shape:** `(batch, 32)`

### Couche 3: Dense(16, relu)

```python
Dense(16, activation='relu')
```

**Couche fully-connected:**
- Combine les 32 valeurs du GRU
- Crée 16 nouvelles features
- `relu = max(0, x)` → activation non-linéaire

### Couche 4: Dense(1, linear)

```python
Dense(1, activation='linear')
```

**Sortie finale:**
- 1 neurone = le prix prédit
- `linear` = pas d'activation (régression)

### Architecture complète

```
Input: (30 jours, 24 features)
   ↓
GRU(64) → Dropout → États: (30, 64)
   ↓
GRU(32) → Dropout → État final: (32,)
   ↓
Dense(16, relu) → Features: (16,)
   ↓
Dense(1, linear) → Prédiction: 1 prix
```

**Paramètres totaux:** 27,233
- Beaucoup moins que des modèles modernes (millions)
- Assez pour capturer les patterns de Tesla

---

## 📊 Entraînement

### Loss Function: MSE

```python
loss='mse'  # Mean Squared Error
```

**Formule:**
```
MSE = Moyenne((Prédiction - Réel)²)
```

**Identique à XGBoost!**

### Optimizer: Adam

```python
optimizer=Adam(learning_rate=0.001)
```

**Adam = Adaptive Moment Estimation**
- Ajuste automatiquement le taux d'apprentissage pour chaque paramètre
- Plus sophistiqué que SGD (Stochastic Gradient Descent)
- `learning_rate=0.001` = valeur standard

### Callbacks

#### EarlyStopping
```python
EarlyStopping(monitor='val_loss', patience=10)
```

**Fonctionnement:**
1. À chaque epoch, calcule `val_loss` (erreur sur validation)
2. Si `val_loss` ne s'améliore pas pendant 10 epochs → STOP
3. Restaure les poids du meilleur epoch

**Analogie avec XGBoost:**
Comme XGBoost arrête d'ajouter des arbres si ça n'améliore plus.

#### ModelCheckpoint
```python
ModelCheckpoint('gru_tsla_best.h5', monitor='val_loss', save_best_only=True)
```

**Sauvegarde automatique:**
- À chaque amélioration de `val_loss` → sauvegarde le modèle
- Garde uniquement le meilleur

---

## 📈 Résultats obtenus

### Métriques

```
GRU  → R²=0.5926 | RMSE=$37.10 | MAE=$26.34 | Gap=0.3854
XGB  → R²=0.8309 | RMSE=$23.90 | MAE=$18.15 | Gap=0.1659
```

### Analyse

#### R² = 0.5926
- Le modèle explique ~59% de la variance
- **Moins bon que XGBoost (83%)**
- Pas terrible mais acceptable pour un premier essai

#### RMSE = $37.10
- Erreur moyenne de ~$37
- **Moins bon que XGBoost ($24)**
- Sur Tesla à $250, ça fait ~15% d'erreur

#### Gap = 0.3854
- **PROBLÈME: Overfitting élevé**
- R² train = 0.9780 (98% sur train!)
- R² test = 0.5926 (59% sur test)
- Le modèle "mémorise" le train mais généralise mal

### Pourquoi le GRU est moins bon?

1. **Overfitting:**
   - Réseau trop complexe pour la quantité de données
   - 27,233 paramètres vs 1,963 séquences train

2. **Marchés financiers:**
   - Très bruités et chaotiques
   - Les RNNs ont du mal avec les changements brusques
   - XGBoost gère mieux les non-linéarités locales

3. **Séquences courtes:**
   - 30 jours peut-être insuffisant pour capturer les cycles longs
   - Ou trop long (introduit du bruit)

---

## 🔧 Pistes d'amélioration

### 1. Réduire l'overfitting

**Augmenter le Dropout:**
```python
Dropout(0.3)  # au lieu de 0.2
Dropout(0.4)  # encore plus agressif
```

**Réduire la complexité:**
```python
GRU(32, ...)  # au lieu de 64
GRU(16, ...)  # au lieu de 32
```

**Ajouter de la régularisation L2:**
```python
from tensorflow.keras.regularizers import l2
GRU(64, kernel_regularizer=l2(0.01), ...)
```

### 2. Ajuster la longueur de séquence

**Tester différentes longueurs:**
```python
SEQUENCE_LENGTH = 20  # Plus court
SEQUENCE_LENGTH = 60  # Plus long
```

**Trade-off:**
- Court (10-20): Moins d'overfitting, perd patterns longs
- Long (60-90): Capture cycles longs, risque overfitting

### 3. Utiliser LSTM au lieu de GRU

**LSTM = Long Short-Term Memory**
- Plus complexe que GRU
- Meilleur pour dépendances très longues
- Plus lent à entraîner

```python
from tensorflow.keras.layers import LSTM
model.add(LSTM(64, return_sequences=True, ...))
```

### 4. Augmenter les données

**Plus de données = moins d'overfitting**
- Télécharger depuis 2010 (15 ans au lieu de 10)
- Ajouter d'autres actions similaires (NVDA, AMD)

### 5. Ensembling

**Combiner GRU + XGBoost:**
```python
prediction_finale = 0.3 * prediction_gru + 0.7 * prediction_xgb
```

---

## 📚 Concepts clés pour l'évaluation

### Pourquoi GRU pour séries temporelles?

**Mémoire temporelle:**
- Les prix d'actions ont une dépendance temporelle
- Le prix d'aujourd'hui dépend de hier, avant-hier, etc.
- GRU "se souvient" de ces dépendances via son état caché

**Exemple:**
```
Tendance haussière sur 20 jours → GRU apprend "momentum positif"
Forte chute récente → GRU ajuste sa prédiction à la baisse
```

### GRU vs LSTM

| Aspect | GRU | LSTM |
|--------|-----|------|
| Complexité | Simpler (2 portes) | Plus complexe (3 portes) |
| Paramètres | Moins | Plus |
| Vitesse | Plus rapide | Plus lent |
| Performance | Souvent équivalente | Meilleur sur séquences très longues |

**Pour Tesla (30 jours), GRU suffit.**

### Normalisation: pourquoi essentielle?

**Sans normalisation:**
```
Prix: 250
Volume: 100,000,000
Volatility: 0.02

Gradient du réseau → dominé par Volume (énorme)
→ Apprentissage inefficace
```

**Avec normalisation:**
```
Prix normalisé: -0.5
Volume normalisé: 1.2
Volatility normalisé: -0.8

Gradient équilibré → Apprentissage efficace
```

### EarlyStopping: pourquoi important?

**Sans EarlyStopping:**
```
Epoch 1-30: val_loss diminue
Epoch 31-50: val_loss stagne
Epoch 51-100: val_loss augmente (overfitting!)
→ Perte de temps + overfitting
```

**Avec EarlyStopping (patience=10):**
```
Epoch 1-30: val_loss diminue
Epoch 31-40: val_loss stagne (patience compteur: 1-10)
Epoch 40: STOP → Restaure epoch 30
→ Économise temps + évite overfitting
```

---

## 🎯 Conclusion

### Forces du GRU

✅ Capture dépendances temporelles (tendances, momentum)
✅ Architecture claire et interprétable
✅ Fonctionne (R²=0.59 pas catastrophique)

### Faiblesses observées

❌ Overfitting élevé (gap=0.38)
❌ Moins performant que XGBoost
❌ Nécessite beaucoup de tuning

### Quand utiliser GRU vs XGBoost?

**Utiliser XGBoost si:**
- Données tabulaires avec features engineered
- Besoin de performance maximale
- Interprétabilité importante (feature importance)

**Utiliser GRU si:**
- Séquences naturelles (texte, audio, vidéo)
- Patterns temporels complexes
- Pas le temps de faire du feature engineering

**Pour Tesla:**
XGBoost gagne car features bien choisies (MA, EMA, MACD, etc.) + moins d'overfitting.

---

## 📖 Ressources

- [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)
- [Keras GRU Documentation](https://keras.io/api/layers/recurrent_layers/gru/)
- [Time Series Forecasting with Deep Learning](https://machinelearningmastery.com/time-series-forecasting-with-deep-learning/)
