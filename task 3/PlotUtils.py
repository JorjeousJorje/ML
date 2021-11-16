import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from numpy import ndarray



def plot_mnist_image(ax, image: ndarray, cmap=None, title: str=None, label: str=None):
    ax.set_axis_off()
    showed_image = ax.imshow(image.reshape(28, 28), cmap=cmap, interpolation="nearest")
    
    if label is not None:
        ax.text(1, 3, label, bbox={'facecolor': 'white', 'pad': 5})
        
    if title is not None:
        ax.set_title(title)
    
    return showed_image

def plot_mnist_samples(X: ndarray, y: ndarray, cmap=None, rows: int=1, cols: int=5, figsize: tuple=(10, 3)):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, constrained_layout=True)

    for ax, image, label in zip(axes.flat, X, y):
        showed_image = plot_mnist_image(ax=ax, image=image, label=str(label), cmap=cmap)
    
    if axes.ndim > 1:
        fig.colorbar(showed_image, ax=axes[:, -1])
    else:
        fig.colorbar(showed_image, ax=axes)
    plt.show()
    

def plot_different_scalers_output(y: ndarray, distributions: list, cmap=None, samples_num: int=5, figsize: tuple=(10, 20)):
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(nrows=len(distributions), ncols=1)
    
    for subfig, (title, images) in zip(subfigs, distributions):
        subfig.suptitle(title)
        axs = subfig.subplots(nrows=1, ncols=samples_num)
        
        for ax, image, label in zip(axs, images[: samples_num], y[: samples_num]):
            showed_image = plot_mnist_image(ax, image, label=str(label), cmap=cmap)
            
        subfig.colorbar(showed_image, ax=ax, orientation='vertical', extend='both')
        
    plt.show()
    
def plot_wrong_predictions(X: ndarray, y: ndarray, preds: ndarray, num_samples: int=5):
    fig, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(10, 3))
    
    wrong_preds_indexes = np.where(preds != y)[0]
    
    X_plot: ndarray = X[wrong_preds_indexes][: num_samples]
    preds_plot: ndarray = preds[wrong_preds_indexes][: num_samples]
    
    plt.suptitle("Wrong predictions of the classifier")
    for ax, img, label in zip(axes, X_plot, preds_plot):
        plot_mnist_image(ax, img, label=str(label))
    
    plt.tight_layout()
    plt.show()
    
    
def plot_embedding(tsne_result, y, figsize: tuple = (10, 10)):
    
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1, figsize=figsize)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    
def plot_support_vectors(svc_estimator, vector_shape: tuple, classes_names, num_sampels=7, figsize: tuple=(10, 20)):
    classes_count = svc_estimator.classes_.shape[0]
    n_supports = svc_estimator.n_support_
    vectors = svc_estimator.support_vectors_
    
    print(f"Support vectors count: {vectors.shape[0]}")
    print(f"Support vectors count for each class: {n_supports}")
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(nrows=classes_count, ncols=1)
    
    for i, (subfig, name) in enumerate(zip(subfigs, classes_names)):
        slice_from = np.sum(n_supports[:i])
        class_vectors = vectors[slice_from: slice_from + num_sampels]
        axs = subfig.subplots(nrows=1, ncols=num_sampels)
        
        subfig.suptitle(f"Support vectors for {name} class")
        for ax, vector in zip(axs, class_vectors):
            ax.set_axis_off()
            ax.imshow(vector.reshape(vector_shape), interpolation="nearest")
    plt.show()