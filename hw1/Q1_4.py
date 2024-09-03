import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,  random_state=42)

model = LogisticRegression(penalty=None)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
v=[print(f"True y = {y_test[i]}, Predicted y = {y_pred[i]}") for i in range(len(y_test))]
print(f'Train score = {model.score(X_train, y_train)}')
print(f'Test score = {model.score(X_test, y_test)}')

model = LogisticRegression(penalty='l2')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
v=[print(f"True y = {y_test[i]}, Predicted y = {y_pred[i]}") for i in range(len(y_test))]
print(f'Train score = {model.score(X_train, y_train)}')
print(f'Test score = {model.score(X_test, y_test)}')