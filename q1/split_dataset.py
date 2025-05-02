import numpy as np

def train_test_split(x, y, test_ratio=0.2, seed=42):
    """
    Splits data into train and test sets.

    :param x: Feature matrix
    :param y: Label vector
    :param test_ratio: Fraction of data to be used for testing
    :param seed: Random seed for reproducibility
    :return: x_train, x_test, y_train, y_test
    """
    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    test_size = int(len(x) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test

