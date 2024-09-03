import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
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
        m = X.shape[0]
        
        d_output = self.output_output - y
        d_weights_hidden_output = np.dot(self.hidden_output.T, d_output) / m
        d_bias_output = np.sum(d_output, axis=0, keepdims=True) / m
        
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * (self.hidden_output * (1 - self.hidden_output))
        d_weights_input_hidden = np.dot(X.T, d_hidden) / m
        d_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True) / m
        
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_hidden -= learning_rate * d_bias_hidden
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_output -= learning_rate * d_bias_output

    def accuracy(self, X, y):
        predictions = np.argmax(self.forward(X), axis=1)
        accuracy = np.mean(predictions == np.argmax(y, axis=1))
        return accuracy

mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target

X_normalized = X / 255.0

num_classes = 10
y_one_hot = np.eye(num_classes)[y.astype(int)]

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_one_hot, test_size=0.2, random_state=42)

X_train = X_train.to_numpy()

input_size = 784
hidden_size = 100
output_size = 10

model = NeuralNetwork(input_size, hidden_size, output_size)

epochs = 10
learning_rate = 0.01
batch_size = 32
num_batches = X_train.shape[0] // batch_size

for epoch in range(epochs):
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]
        
        model.forward(X_batch)
        
        model.backward(X_batch, y_batch, learning_rate)
    
    train_accuracy = model.accuracy(X_train, y_train)
    test_accuracy = model.accuracy(X_test, y_test)
    print("Epoch {}/{} - Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(epoch + 1, epochs, train_accuracy, test_accuracy))

###########################################################################################################
########################################## project 4 ######################################################
###########################################################################################################

#الف
import numpy as np

def calculate_class_accuracy(predictions, true_labels):
    class_accuracies = {}
    unique_classes = np.unique(np.argmax(true_labels, axis=1))

    for cls in unique_classes:
        cls_indices = np.argmax(true_labels, axis=1) == cls
        cls_predictions = predictions[cls_indices]
        cls_true_labels = true_labels[cls_indices]
        
        correct_predictions = np.sum(np.argmax(cls_predictions, axis=1) == np.argmax(cls_true_labels, axis=1))
        class_accuracy = correct_predictions / len(cls_indices)
        class_accuracies[cls] = class_accuracy
    
    overall_accuracy = np.mean([np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1)])
    return class_accuracies, overall_accuracy

# فرض می‌کنیم X_test و y_test داده‌های تست هستند که از قبل بارگذاری شده‌اند
test_predictions = model.forward(X_test)
class_accuracies, overall_accuracy = calculate_class_accuracy(test_predictions, y_test)

print("Class Accuracies:")
for cls, acc in class_accuracies.items():
    print(f"Class {cls}: {acc:.4f}")
print(f"Overall Accuracy: {overall_accuracy:.4f}")

#ب
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics(predictions, true_labels):
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(true_labels, axis=1)
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    return precision, recall, f1_score

# فرض می‌کنیم X_test و y_test داده‌های تست هستند که از قبل بارگذاری شده‌اند
test_predictions = model.forward(X_test)
precision, recall, f1_score = calculate_metrics(test_predictions, y_test)

# چاپ نتایج برای هر کلاس
for cls in range(len(precision)):
    print(f"Class {cls} - Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}, F1-Score: {f1_score[cls]:.4f}")

#ج
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

test_predictions = model.forward(X_test)

# تابع برای رسم نمودار ROC برای هر کلاس
def plot_multiclass_roc(y_test, y_score, n_classes):
    # محاسبه ROC curve و ROC area برای هر کلاس
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # رسم نمودار ROC برای هر کلاس
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-Class')
    plt.legend(loc="lower right")
    plt.show()

# داده‌های y_test را برای کلاس‌های مورد نیاز یک‌سان می‌کنیم
y_test_bin = label_binarize(y_test, classes=[i for i in range(10)])
plot_multiclass_roc(y_test_bin, test_predictions, 10)


#د
from sklearn.metrics import jaccard_score

# Assuming 'model' is your trained neural network and 'X_test' is your test dataset
test_predictions = model.forward(X_test)
y_pred = np.argmax(test_predictions, axis=1)  # Converting probabilities to class labels
y_true = np.argmax(y_test, axis=1)            # Assuming y_test is already one-hot encoded

# Calculate Jaccard Score
# Using 'macro' average to treat all classes equally
jaccard = jaccard_score(y_true, y_pred, average='macro')

print(f"Jaccard Score (Macro Average): {jaccard:.4f}")
