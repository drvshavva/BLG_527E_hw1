import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pd

def plot_dataset(X, y, mus, sigmas):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. SCATTER PLOT
    axs[0, 0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], alpha=0.5, label='Class 0')
    axs[0, 0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.5, label='Class 1')
    axs[0, 0].set_title('Scatter Plot')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. CONTOUR PLOT
    x, y_ = np.mgrid[
        np.min(X[:, 0]) - 1:np.max(X[:, 0]) + 1:0.1,
        np.min(X[:, 1]) - 1:np.max(X[:, 1]) + 1:0.1
    ]
    pos = np.dstack((x, y_))
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        rv = multivariate_normal(mean=mu, cov=sigma)
        axs[0, 1].contour(x, y_, rv.pdf(pos), levels=5, colors='k' if i == 0 else 'r')
    axs[0, 1].set_title('Contour Plot (Normal Distribution)')
    axs[0, 1].grid(True)

    # 3. HISTOGRAMS
    for i in range(2):
        axs[1, 0].hist(X[y == 0][:, i], bins=30, alpha=0.5, label=f'Class 0 - Feature {i+1}')
        axs[1, 0].hist(X[y == 1][:, i], bins=30, alpha=0.5, label=f'Class 1 - Feature {i+1}')
    axs[1, 0].set_title('Histograms')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. KDE PLOT (Seaborn needs Pandas DataFrame)
    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    df["Class"] = y
    sns.kdeplot(data=df, x="Feature 1", y="Feature 2", hue="Class", fill=True, alpha=0.4, ax=axs[1, 1])
    axs[1, 1].set_title("KDE Plot")

    plt.tight_layout()
    plt.show()
