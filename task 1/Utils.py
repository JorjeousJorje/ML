import matplotlib.pyplot as plt
from itertools import product
import numpy as np


def plot_features(X: np.array, Y: np.array, f_names, indexes=None):

    plt.figure(figsize=(15, 15), dpi=100)
    num_features = X.shape[1]

    for i, (f1, f2) in enumerate(product(range(num_features), range(num_features))):
        plt.subplot(num_features, num_features, 1 + i)

        if f1 == f2:
            plt.text(0.25, 0.5, f_names[f1])
        else:
            plt.scatter(X[:, f1], X[:, f2], c=Y)
            if (indexes != None):
                plt.scatter(X[indexes, f1], X[indexes, f2], c='r', marker='x')

    plt.tight_layout()

def wrong_predicts(predicts, gt):
    wrong_preds =[]
    for index in range(len(predicts)):
        if predicts[index] != gt[index]:
            wrong_preds.append(index)
    return wrong_preds  

def plot_points(x: np.array, y: np.array, сlasses: np.array, marked_indexes = None, axes_names:tuple[str, str] = None):
    plt.scatter(x, y, c=сlasses)
    
    if axes_names is None:
        axes_names = ("x", "y")

    if marked_indexes is not None:
        plt.scatter(x[marked_indexes], y[marked_indexes], c='r', marker='x')

    plt.xlabel(axes_names[0])
    plt.ylabel(axes_names[1])   