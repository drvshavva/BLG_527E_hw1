import numpy as np
import matplotlib.pyplot as plt


def qda_predict(X, mus, sigmas, priors=None):
    """
    Manually predicts class labels using QDA (class-specific covariances)

    :param X: Data points to classify (N x D)
    :param mus: List of mean vectors per class
    :param sigmas: List of covariance matrices per class
    :param priors: List of class priors (optional); if None, uniform assumed
    :return: predicted labels
    """
    n_classes = len(mus)
    n_samples = X.shape[0]
    scores = np.zeros((n_samples, n_classes))

    if priors is None:
        priors = [1 / n_classes] * n_classes

    for c in range(n_classes):
        sigma_inv = np.linalg.inv(sigmas[c])  # Sınıf c'nin kovaryans matrisinin tersi
        sigma_det = np.linalg.det(sigmas[c])  # Sınıf c'nin kovaryans matrisinin determinantı
        for i in range(n_samples):
            x = X[i]  # i. veri örneği
            diff = x - mus[c]  # Ortalamadan fark vektörü (x - μ_c)
            term = -0.5 * np.dot(np.dot(diff.T, sigma_inv), diff)
            # Mahalanobis mesafesi: -0.5 * (x - μ_c)^T Σ_c^-1 (x - μ_c)
            normalization = -0.5 * np.log(sigma_det + 1e-8)  # avoid log(0)
            prior_term = np.log(priors[c])
            scores[i, c] = term + normalization + prior_term

    return np.argmax(scores, axis=1)


def plot_decision_boundary(X_train, y_train, mus, sigmas):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = qda_predict(grid, mus, sigmas)
    zz = zz.reshape(xx.shape)

    # Karar sınırı + noktalar
    plt.contourf(xx, yy, zz, alpha=0.3, levels=[-0.1, 0.5, 1.1], colors=['#FFAAAA', '#AAAAFF'])
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Class 0", alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Class 1", alpha=0.6)
    plt.title("QDA Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
