import numpy as np
import matplotlib.pyplot as plt


def lda_predict(X, mus, shared_sigma, priors=None):
    """
    Predicts class labels using Linear Discriminant Analysis (shared covariance).

    :param X: Data points (N x D)
    :param mus: List of mean vectors per class
    :param shared_sigma: Shared covariance matrix (D x D)
    :param priors: List of class priors (optional), defaults to uniform
    :return: predicted class labels (N,)
    """
    n_classes = len(mus)
    n_samples = X.shape[0]
    scores = np.zeros((n_samples, n_classes))

    sigma_inv = np.linalg.inv(shared_sigma)

    if priors is None:
        priors = [1 / n_classes] * n_classes

    for k in range(n_classes):
        mu_k = mus[k]
        for i in range(n_samples):
            x = X[i]
            term1 = np.dot(np.dot(x.T, sigma_inv), mu_k)
            term2 = -0.5 * np.dot(np.dot(mu_k.T, sigma_inv), mu_k)
            term3 = np.log(priors[k])
            scores[i, k] = term1 + term2 + term3

    return np.argmax(scores, axis=1)


def plot_lda_decision_boundary(X_train, y_train, mus, shared_sigma):
    """
    Plots the decision boundary for LDA classifier.

    :param X_train: Training data (N x 2)
    :param y_train: Training labels (N,)
    :param mus: Class mean vectors
    :param shared_sigma: Shared covariance matrix
    """
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    zz = lda_predict(grid, mus, shared_sigma)
    zz = zz.reshape(xx.shape)

    plt.contourf(xx, yy, zz, levels=[-0.1, 0.5, 1.1], alpha=0.3, colors=['#FFAAAA', '#AAAAFF'])
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Class 0', alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Class 1', alpha=0.6)
    plt.title("LDA Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
