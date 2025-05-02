import numpy as np


def generate_data(mus: list, sigmas: list, number_of_class: int, number_of_samples: int):
    """
    Generates synthetic data for a multi-class Gaussian dataset.

    :param mus: List of mean vectors for each class
    :param sigmas: List of covariance matrices for each class
    :param number_of_class: Number of classes
    :param number_of_samples: Number of samples per class

    :return: (x, y) where x is feature array and y is label array
    """

    assert len(mus) == len(sigmas) == number_of_class, \
        "There must be the same number of mus, sigmas, and number_of_class"

    x, y = list(), list()

    for class_index in range(number_of_class):
        samples = np.random.multivariate_normal(
            mean=mus[class_index],
            cov=sigmas[class_index],
            size=number_of_samples
        )
        x.append(samples)
        y.append(np.full(number_of_samples, class_index))

    x = np.vstack(x)
    y = np.concatenate(y)

    return x, y


