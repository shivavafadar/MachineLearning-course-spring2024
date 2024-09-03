import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

sklearn.datasets.data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(sklearn.datasets.data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8,  random_state=42)

def normalization(data):
  min_vals = np.min(data, axis=0)  
  max_vals = np.max(data, axis=0)  
  normalized_data = (data - min_vals) / (max_vals - min_vals)
  return normalized_data

def mini_batch_with_regularization_train(x, y, theta, batch_size=32, learning_rate=.1, epochs=1000, lambda_val=.0001):
  m = len(y)
  num_batches = m // batch_size
  if m % batch_size != 0:
    num_batches += 1

  x = normalization(x)
  x0 = np.ones((len(x),1))
  x= np.concatenate((x0, x), axis=1)

  for epoch in range(epochs):
    permutation  = np.random.permutation(m)
    shuffeled_x = x[permutation]
    shuffeled_y = y[permutation]

    for i in range(num_batches):
      start_idx = i * batch_size
      end_idx = min((i + 1) * batch_size, m)
      x_batch = shuffeled_x[start_idx:end_idx]
      y_batch = shuffeled_y[start_idx:end_idx]

      predictions = np.dot(x_batch, theta)
      errors = predictions - y_batch

      gradient = (np.dot(x_batch.T, errors) + lambda_val * theta) / len(y_batch)
      gradient[0] -= lambda_val * theta[0]  # Exclude regularization for bias term

      theta -= learning_rate * gradient

    if (epoch + 1) % 10 == 0:
      cost = np.mean((np.dot(x, theta) - y) ** 2) / 2 + (lambda_val / (2 * m)) * np.sum(theta[1:] ** 2)  # Regularized cost
      print(f'Epoch {epoch + 1}/{epochs}, Cost: {cost}')

  return theta

theta = np.zeros(len(x_train[0])+1)
theta_final = mini_batch_with_regularization_train(x_train, y_train, theta)

y_pred = predict(x_test, y_test, theta_final)
y_pred = np.round(y_pred,2)
v=[print(f"True y = {y_test[i]}, Predicted y = {y_pred[i]}") for i in range(len(y_test))]