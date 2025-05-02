import numpy as np
from q3.lda import compute_lda_components, plot_lda, plot_lda_projection
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent

def run():
    data = np.loadtxt(f"{project_path}/data.txt", delimiter=",")
    x = data[:, :-1]  # features
    y = data[:, -1]  # class labels
    X_lda, components = compute_lda_components(x, y, n_components=2)
    plot_lda(X_lda, y)
    plot_lda_projection(X_lda, y)


if __name__ == '__main__':
    run()
