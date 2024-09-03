import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
sklearn.datasets.data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(sklearn.datasets.data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, train_size=0.8,  random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)

print("Train:", train_mse)
print("Test:", test_mse)

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, train_size=0.8,  random_state=42)
model = Ridge(alpha=1.5)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)