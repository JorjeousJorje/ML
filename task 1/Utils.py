import matplotlib.pyplot as plt
from itertools import product
import numpy as np


def plot_features_data(X: np.ndarray, y: np.ndarray, features_names) -> None:
    with plt.style.context(['science']):
        num_features = X.shape[1]
        grid = product(range(num_features), range(num_features))
        
        for i, (name1, name2) in enumerate(grid):
            plt.subplot(num_features, num_features, 1 + i)

            if name1 == name2:
                plt.text(0.25, 0.5, features_names[name1])
            else:
                plt.scatter(X[:, name1], X[:, name2], c=y)
                plt.xlabel(features_names[name1])
                plt.ylabel(features_names[name2])

    plt.tight_layout()
    
def plot_points(X: np.ndarray, y: np.ndarray, classes: np.ndarray, marked_indexes=None, axes_names:tuple[str, str]=None):
    with plt.style.context(['science']):
        plt.scatter(X, y, c=classes)
        if axes_names is None:
            axes_names = ("x", "y")

        if not marked_indexes is None:
            plt.scatter(X[marked_indexes], y[marked_indexes], c='r', marker='x')

        plt.xlabel(axes_names[0])
        plt.ylabel(axes_names[1])   