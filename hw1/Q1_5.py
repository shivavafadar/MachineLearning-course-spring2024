import numpy as np
import pandas as pd
import numpy as np
import sklearn

def normal_equation_with_regularization(X, y, lambda_reg):
    m, n = X.shape
    regularization_matrix = lambda_reg * np.identity(n)
    theta = np.linalg.inv(X.T.dot(X) + regularization_matrix).dot(X.T).dot(y)
    return theta

np.random.seed(0)
X = 2 * np.random.rand(100, 3)
y = 4 + np.dot(X, np.array([3, 1.5, 2])) + np.random.randn(100)

X_b = np.c_[np.ones((100, 1)), X]

lambda_reg = 0.3

theta = normal_equation_with_regularization(X_b, y, lambda_reg)
print("Parameters with regularization:", theta)
