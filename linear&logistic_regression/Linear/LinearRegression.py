import numpy as np
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, alpha=0.01):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.loss_step = []

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, alpha = 0.01):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.alpha = alpha
        self.loss_step = []

        num_v_sam = int(0.1 * X.shape[0])
        X_val, y_val = X[:num_v_sam], y[:num_v_sam]
        X_train, y_train = X[num_v_sam:], y[num_v_sam:]

        num_samp, num_feat = X_train.shape
        num_out = y_train.shape[1]
        
        # TODO: Initialize the weights and bias based on the shape of X and y.

        self.weights = np.zeros((num_feat, num_out))
        self.bias = np.zeros(num_out)
        
        best_wt = self.weights.copy
        best_bi = self.bias.copy()
        bl = float('inf')
        inc = 0

        # TODO: Implement the training loop.
        for i in range(self.max_epochs):
            id = np.random.permutation(num_samp)
            X_sh, y_sh = X_train[id], y_train[id]

            for j in range(0, num_samp, self.batch_size):
                X_bat = X_sh[j:j + self.batch_size]
                y_bat = y_sh[j:j + self.batch_size]
                
                y_pred = self.predict(X_bat)
                error = y_pred - y_bat
                loss = self.score(X_bat, y_bat) + self.regularization * np.sum(self.weights ** 2)
                self.loss_step.append(loss)           #loss vs step plot

                gw = (X_bat.T @ error + 2 * self.regularization * self.weights) / self.batch_size
                gb = (1/self.batch_size) * np.sum(y_pred - y_bat, axis=0)

                self.weights -= alpha * gw
                self.bias -= alpha * gb

            #y_pred_val = self.predict(X_val)
            val_loss = self.score(X_val, y_val)

            if val_loss < bl:
                bl = val_loss
                best_wt = self.weights.copy()
                best_bi = self.bias.copy()
                inc = 0
            else:
                inc += 1
                if inc >= self.patience:
                    break

        self.weights = best_wt
        self.bias = best_bi
            

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)
   
    def save(self, fp):
        np.savez(fp, weights = self.weights, bias = self.bias)

    def load(self, fp):
        data = np.load(fp)
        self.weights = data['weights']
        self.bias = data['bias']

