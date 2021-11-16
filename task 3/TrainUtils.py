import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score

from PlotUtils import plot_wrong_predictions


def plot_confusion_matrix(estimator, predictions, y_test, figsize: tuple=(15, 12)):
    cm = confusion_matrix(y_test, predictions, labels=estimator.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_)
    _, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax)

def cross_validate_wrapper(clf, X: ndarray, y: ndarray, train_size=5000, test_size: int=5000, scoring='accuracy'):
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)
    cv_results = cross_validate(clf, X_train, y_train, scoring=scoring)
    scores = cv_results['test_score']
    print("Crossvalidating...")
    print(f"> Current metric is: {scoring}")
    print(f"> Mean metric is: {sum(scores) / len(scores)}")

def fit_predict_wrapper(clf, X: ndarray, y: ndarray, train_size=5000, test_size=5000, metric=accuracy_score):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = metric(y_test, predictions)
    print(f"Accuracy score: {accuracy}")

def gridsearch_wrapper(pipe_clf, X: ndarray, y: ndarray, train_size=5000, test_size=5000):
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)
    pipe_clf.fit(X_train, y_train)
    
    best_params = pipe_clf[-1].best_params_
    best_score = pipe_clf[-1].best_score_
    
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")
    return best_params
    