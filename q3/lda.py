import random

import numpy as np
from matplotlib import pyplot as plt


def compute_class_means(X, y):
    classes = np.unique(y)
    means = {}
    for c in classes:
        class_samples = X[y == c]
        n_samples = class_samples.shape[0]
        mu = np.sum(class_samples, axis=0) / n_samples
        means[c] = mu
    return means


def compute_lda_components(X, y, n_components=2):
    n_features = X.shape[1]
    classes = np.unique(y)
    overall_mean = np.sum(X, axis=0) / X.shape[0]

    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))

    # Class mean vectors
    means = {}
    for c in classes:
        class_data = X[y == c]
        mu_k = np.sum(class_data, axis=0) / class_data.shape[0]
        means[c] = mu_k

        # Within-class scatter
        for x in class_data:
            diff = (x - mu_k).reshape(-1, 1)
            S_W += np.dot(diff, diff.T)

        # Between-class scatter
        mean_diff = (mu_k - overall_mean).reshape(-1, 1)
        S_B += class_data.shape[0] * np.dot(mean_diff, mean_diff.T)

    S_W_inv = np.linalg.pinv(S_W)
    eigvals, eigvecs = np.linalg.eig(S_W_inv @ S_B)

    # Sort by descending eigenvalue
    sorted_indices = np.argsort(eigvals)[::-1]
    top_components = eigvecs[:, sorted_indices[:n_components]]

    # Project data
    X_lda = X @ top_components.real

    return X_lda, top_components.real


def plot_lda_projection(X_lda, y, annotate_count=200):
    plt.figure(figsize=(12, 8))

    plt.scatter(X_lda[:, 0], X_lda[:, 1], c='gray', alpha=0.1, s=10)

    indices = random.sample(range(len(y)), annotate_count)
    for idx in indices:
        plt.annotate(str(int(y[idx])), (X_lda[idx, 0], X_lda[idx, 1]),
                     fontsize=8, color='black', alpha=0.7)

    plt.title("LDA Projection to 2D")
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_lda(X_pca, y, annotate_count=200):
    plt.figure(figsize=(12, 8))

    # Scatter all points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)

    # Annotate 200 random points
    indices = random.sample(range(len(y)), annotate_count)
    for i in indices:
        plt.annotate(str(int(y[i])), (X_pca[i, 0], X_pca[i, 1]), fontsize=7, alpha=0.7)

    plt.title("LDA Projection to 2D")
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
