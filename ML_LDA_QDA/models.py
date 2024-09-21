import numpy as np


class LDAModel:
    def __init__(self):
        self.class_means = None
        self.covariance_matrix = None
        self.inv_covariance_matrix = None
        self.class_priors = None

    def fit(self, X, y):
        X = np.array(X).reshape(len(X), -1)
        y = np.array(y)
        unique_classes = np.unique(y)
        self.class_means = np.array(
            [np.mean(X[y == u], axis=0) for u in unique_classes]
        )
        centered_data = X - self.class_means[y]
        self.covariance_matrix = np.dot(centered_data.T, centered_data) / X.shape[0]
        self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        class_counts = np.bincount(y)
        self.class_priors = class_counts / len(y)

    def predict(self, X):
        X = np.array(X).reshape(len(X), -1)
        diffs = X[:, np.newaxis, :] - self.class_means
        diffs_inv_cov = np.matmul(diffs, self.inv_covariance_matrix)
        squared_distances = np.sum(diffs * diffs_inv_cov, axis=2)
        log_probs = np.log(self.class_priors) - 0.5 * squared_distances
        y_pred = np.argmax(log_probs, axis=1)
        return y_pred


class QDAModel:
    def __init__(self):
        self.class_means = None
        self.class_cov_matrices = None
        self.class_inv_cov_matrices = None
        self.class_priors = None
        self.unique_classes = None

    def fit(self, X, y):
        X = np.array(X).reshape(len(X), -1)
        y = np.array(y)
        self.unique_classes = np.unique(y)
        self.class_means = np.array(
            [np.mean(X[y == u], axis=0) for u in self.unique_classes]
        )
        self.class_cov_matrices = [np.cov(X[y == u].T) for u in self.unique_classes]
        self.class_inv_cov_matrices = [
            np.linalg.inv(cov) for cov in self.class_cov_matrices
        ]
        class_counts = np.bincount(y)
        self.class_priors = class_counts / len(y)

    def predict(self, X):
        X = np.array(X).reshape(len(X), -1)
        y_pred = np.zeros(len(X), dtype=int)
        for i, test_sample in enumerate(X):
            scores = np.zeros(len(self.unique_classes))
            for u, class_mean in enumerate(self.class_means):
                diff = test_sample - class_mean
                scores[u] = np.log(self.class_priors[u]) - 0.5 * np.sum(
                    np.dot(diff.T, np.dot(self.class_inv_cov_matrices[u], diff))
                )
            y_pred[i] = np.argmax(scores)
        return y_pred


class GaussianNBModel:
    def __init__(self):
        self.class_means = None
        self.class_variances = None
        self.class_priors = None

    def fit(self, X, y):
        X = np.array(X).reshape(len(X), -1)
        y = np.array(y)
        self.unique_classes = np.unique(y)
        self.class_means = np.array(
            [np.mean(X[y == u], axis=0) for u in self.unique_classes]
        )
        self.class_variances = np.array(
            [np.var(X[y == u], axis=0) for u in self.unique_classes]
        )
        class_counts = np.bincount(y)
        self.class_priors = class_counts / len(y)

    def predict(self, X):
        X = np.array(X).reshape(len(X), -1)
        y_pred = []
        for x in X:
            class_scores = []
            for mean, variance in zip(self.class_means, self.class_variances):
                log_likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * variance) + ((x - mean) ** 2) / variance
                )
                class_scores.append(log_likelihood)
            y_pred.append(np.argmax(class_scores))
        return np.array(y_pred)
