import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


class LogisticRegression:
    def __init__(self, alpha=0.01, max_epochs=1000, tolerance=1e-4):
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.val_loss = None

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.max_epochs):
            y_pred = self.predict(X)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.loss_history.append(loss)

            if X_val is not None and y_val is not None:
                y_pred_val = self.sigmoid(np.dot(X_val, self.weights) + self.bias)
                val_loss = -np.mean(y_val * np.log(y_pred_val) + (1 - y_val) * np.log(1 - y_pred_val))
                self.val_loss = val_loss

            gw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            gb = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.alpha * gw
            self.bias -= self.alpha * gb

            if np.linalg.norm(gw) < self.tolerance:
                break

    def predict(self, X):
        return np.round(self.sigmoid(np.dot(X, self.weights) + self.bias))

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def save(self, fp):
        np.savez(fp, weights = self.weights, bias = self.bias)

    def load(self, fp):
        data = np.load(fp)
        self.weights = data['weights']
        self.bias = data['bias']









