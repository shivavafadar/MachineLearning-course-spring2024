import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LinearDiscriminantAnalysis:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_priors = {}
        self.means = {}
        self.shared_covariance = None

        for c in self.classes:
            X_class = X_train[y_train == c]
            self.class_priors[c] = len(X_class) / len(X_train)
            self.means[c] = np.mean(X_class, axis=0)

        self.shared_covariance = np.zeros((X_train.shape[1], X_train.shape[1]))
        for c in self.classes:
            X_class = X_train[y_train == c]
            diff = X_class - self.means[c]
            self.shared_covariance += np.dot(diff.T, diff) / len(X_train)

    def discriminant_function(self, x, c):
        mean = self.means[c]
        covariance = self.shared_covariance
        prior = self.class_priors[c]

        w = np.dot(np.linalg.inv(covariance), mean)
        w0 = -0.5 * np.dot(np.dot(mean.T, np.linalg.inv(covariance)), mean) + np.log(prior)
        return np.dot(w, x) + w0

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            max_discriminant = float('-inf')
            predicted_class = None
            for c in self.classes:
                discriminant = self.discriminant_function(x, c)
                if discriminant > max_discriminant:
                    max_discriminant = discriminant
                    predicted_class = c
            predictions.append(predicted_class)
        return predictions

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
