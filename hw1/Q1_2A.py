import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

def normalization(X):
  min_vals = np.min(X, axis=0)  # Calculate the minimum values along the columns
  max_vals = np.max(X, axis=0)  # Calculate the maximum values along the columns
  normalized_X = (X - min_vals) / (max_vals - min_vals)
  return normalized_X

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X = normalization(X)
x0 = np.ones((len(X),1))
X = np.concatenate((x0, X), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,  random_state=42)

theta = np.zeros(len(X_train[0]))

def sigmoid(z):
  return 1/(1+np.e**(-z))

def cost(X, y, theta):
  z = np.dot(X, theta)
  cost_0 = np.dot(y.T, np.log(sigmoid(z)))
  cost_1 = np.dot((1-y).T, np.log(1 - sigmoid(z)))
  cost = -(cost_0 + cost_1)/len(y)
  return cost

def fit(X, y, theta, learning_rate=.001, epochs=100):
  for epoch in range(epochs):
    z = np.dot(X, theta)
    gradient = np.dot(X.T, (sigmoid(z) - y))/len(y)
    theta -= learning_rate*gradient
    if (epoch + 1) % 10 == 0:
      print(f'Epoch {epoch + 1}/{epochs}, Cost: {cost(X, y, theta)}')
  return theta

def predict(X, y, theta):
  z = np.dot(X, theta)
  lis = []
  for i in sigmoid(z):
    if i > .5:
      lis.append(1)
    else:
      lis.append(0)
  print(f'Test Cost: {cost(X, y, theta)}')
  return lis

def f1_score(y,y_pred):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score

theta = fit(X_train, y_train, theta)
y_pred = predict(X_test, y_test, theta)
v=[print(f"True y = {y_test[i]}, Predicted y = {y_pred[i]}") for i in range(len(y_test))]
y_pred_train = predict(X_train, y_train, theta)
print(f'f1_score train = {f1_score(y_train, y_pred_train)}')
print(f'f1_score test = {f1_score(y_test, y_pred)}')