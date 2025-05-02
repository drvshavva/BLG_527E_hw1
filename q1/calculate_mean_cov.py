import numpy as np

def compute_class_stats_matrices(x, y, number_of_class):
    """
    Computes mean vectors and covariance matrices for each class manually (no built-in methods).

    :param x:  data features
    :param y: training data labels
    :param number_of_class: number of distinct classes
    :return: (mus, sigmas) where
        - mus is a list of mean vectors per class
        - sigmas is a list of covariance matrices per class
    """
    n_features = x.shape[1]
    mus = []
    sigmas = []

    for class_idx in range(number_of_class):
        class_samples = []
        for i in range(len(y)):
            if y[i] == class_idx:
                class_samples.append(x[i])
        class_samples = np.array(class_samples)
        n_samples = class_samples.shape[0]

        # --- Compute mean vector manually ---
        mu = [0.0] * n_features
        for i in range(n_samples):
            for j in range(n_features):
                mu[j] += class_samples[i][j]
        mu = [v / n_samples for v in mu]

        # --- Compute covariance matrix manually ---
        sigma = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
        for i in range(n_samples):
            for j in range(n_features):
                for k in range(n_features):
                    sigma[j][k] += (class_samples[i][j] - mu[j]) * (class_samples[i][k] - mu[k])
        sigma = [[val / (n_samples - 1) for val in row] for row in sigma]

        mus.append(np.array(mu))
        sigmas.append(np.array(sigma))

    return mus, sigmas


def compute_shared_covariance(x_train, y_train, mus, number_of_class):
    """
    Computes the shared covariance matrix

    :param x_train: Training features (N x D)
    :param y_train: Training labels (N,)
    :param mus: List of mean vectors for each class
    :param number_of_class: Total number of classes
    :return: Shared covariance matrix (D x D)
    """
    n_features = x_train.shape[1]
    sigma = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    total_samples = 0

    for class_index in range(number_of_class):
        class_data = []
        for i in range(len(y_train)):
            if y_train[i] == class_index:
                class_data.append(x_train[i])
        class_data = np.array(class_data)
        mu = mus[class_index]
        n = class_data.shape[0]
        total_samples += n

        for i in range(n):
            for j in range(n_features):
                for k in range(n_features):
                    sigma[j][k] += (class_data[i][j] - mu[j]) * (class_data[i][k] - mu[k])

    sigma = [[val / (total_samples - number_of_class) for val in row] for row in sigma]
    return np.array(sigma)
