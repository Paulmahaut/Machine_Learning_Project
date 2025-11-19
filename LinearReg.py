import itertools
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split #  Split the dataset into "test set" & "train set"
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix


df = pd.read_excel('realestatedataset.xlsx')
# print(df.info())
# print(df.describe(include='all'))

y = df['Y house price of unit area'] #Label (178x1)
x = df[['X6 longitude', 'X2 house age']] # Features (178x12)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 48)


s = StandardScaler()
xs_train = s.fit_transform(x_train) # Normalizing to zero mean unit variance
xs_test = s.fit_transform(x_test)

# Dev
mymodel = LinearRegression()

mymodel.fit(xs_train, y_train)

y_pred = mymodel.predict(xs_test)

print(list(y_test))
print(y_pred)
print(mymodel.coef_, mymodel.intercept_)


# print(f"Mean Error:{mean_absolute_error(y_test, y_pred)}")
# plt.plot(list(y_test),label='Actual',color='blue')
# plt.plot(list(y_pred),label='Predicted',color='red')
# plt.legend()
# plt.show()

features = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 
           'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

results = {}

# On teste toutes les tailles de combinaison (1 à 6)
for r in range(1, len(features) + 1):
    for combo in itertools.combinations(features, r):
        X = df[list(combo)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        results[combo] = mae
        print(f"{combo} -> MAE = {mae:.4f}")

# Trouver la combinaison avec la plus petite MAE
best_combo = min(results, key=results.get)
print("\nLa meilleure combinaison est :", best_combo, "avec MAE =", results[best_combo])