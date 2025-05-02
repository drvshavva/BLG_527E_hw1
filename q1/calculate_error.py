import numpy as np


def classification_error(y_true, y_pred):
    return np.mean(y_true != y_pred)
