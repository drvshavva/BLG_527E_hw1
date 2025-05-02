import random

import numpy as np
from matplotlib import pyplot as plt


def compute_mean(x):
    # compute column-wise mean
    n_samples, n_features = x.shape
    mean = [0.0] * n_features
    for i in range(n_samples):
        for j in range(n_features):
            mean[j] += x[i][j]
    mean = [v / n_samples for v in mean]
    return np.array(mean)


def center_data(x, mean):
    # Subtract the mean from data
    X_centered = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            X_centered[i][j] = x[i][j] - mean[j]
    return X_centered


def compute_covariance_matrix(x):
    # covariance matrix calculation
    n_samples = x.shape[0]
    n_features = x.shape[1]
    cov = [[0.0 for _ in range(n_features)] for _ in range(n_features)]

    for i in range(n_samples):
        for j in range(n_features):
            for k in range(n_features):
                cov[j][k] += x[i][j] * x[i][k]

    cov = [[val / (n_samples - 1) for val in row] for row in cov]
    return np.array(cov)


def pca(x, n_components=2):
    mean = compute_mean(x)
    X_centered = center_data(x, mean)
    cov_matrix = compute_covariance_matrix(X_centered)

    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

    # Sort by eigenvalue magnitude
    sorted_indices = np.argsort(eig_vals)[::-1]
    top_vectors = eig_vecs[:, sorted_indices[:n_components]]

    # Project data
    X_pca = np.dot(X_centered, top_vectors)
    return X_pca, top_vectors


def plot_pca_digits(X_pca, y, annotate_count=200):
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.1, s=10)

    indices = random.sample(range(len(y)), annotate_count)
    for idx in indices:
        plt.annotate(str(int(y[idx])),
                     (X_pca[idx, 0], X_pca[idx, 1]),
                     fontsize=8,
                     color='black',
                     alpha=0.7)

    plt.title("Data after PCA")
    plt.xlabel("First eigenvector")
    plt.ylabel("Second eigenvector")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_pca(X_pca, y, annotate_count=200):
    plt.figure(figsize=(10, 8))

    # Scatter all points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)

    # Annotate 200 random points
    indices = random.sample(range(len(y)), annotate_count)
    for i in indices:
        plt.annotate(str(int(y[i])), (X_pca[i, 0], X_pca[i, 1]), fontsize=7, alpha=0.7)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection to 2D')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
