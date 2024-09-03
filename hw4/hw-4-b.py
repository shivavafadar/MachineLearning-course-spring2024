import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_output = self.softmax(self.output_input)
        return self.output_output
    
    def backward(self, X, y, learning_rate):
        # Forward pass to ensure all activations are updated
        self.forward(X)
        m = X.shape[0]
        
        # Error at output layer
        d_output = self.output_output - y
        d_weights_hidden_output = np.dot(self.hidden_output.T, d_output) / m
        d_bias_output = np.sum(d_output, axis=0, keepdims=True) / m
        
        # Error at hidden layer
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * (self.hidden_output * (1 - self.hidden_output))
        d_weights_input_hidden = np.dot(X.T, d_hidden) / m
        d_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_hidden -= learning_rate * d_bias_hidden
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_output -= learning_rate * d_bias_output

    def accuracy(self, X, y):
        predictions = np.argmax(self.forward(X), axis=1)
        accuracy = np.mean(predictions == np.argmax(y, axis=1))
        return accuracy


# Function to perform PCA
def pca(X, num_components):
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_indices = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_indices]
    eigenvector_subset = sorted_eigenvectors[:, :num_components]
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    return X_reduced

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)
y_one_hot = np.eye(10)[y]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Apply PCA
X_train_pca = pca(X_train, 100)  # Reduce dimension to retain 95% variance, number of components needs adjustment
X_test_pca = pca(X_test, 100)

# Define and train the neural network
input_size = 100  # Adjusted according to PCA
hidden_size = 50
output_size = 10
model = NeuralNetwork(input_size, hidden_size, output_size)

# Training loop
epochs = 10
learning_rate = 0.01
for epoch in range(epochs):
    model.backward(X_train_pca, y_train, learning_rate)
    train_acc = model.accuracy(X_train_pca, y_train)
    test_acc = model.accuracy(X_test_pca, y_test)
    print(f"Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
