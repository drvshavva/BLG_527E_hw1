import numpy as np
from q2.pca import pca, plot_pca, plot_pca_digits
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent

def run():
    data = np.loadtxt(f"{project_path}/data.txt", delimiter=",")
    x = data[:, :-1]  # features
    y = data[:, -1]  # class labels
    X_pca, components = pca(x, n_components=2)
    plot_pca(X_pca, y)
    plot_pca_digits(X_pca, y)


if __name__ == '__main__':
    run()
