import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class UnusedPixelsRemover(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        self.removing_mask: np.ndarray = None
    
    def fit(self, X: np.ndarray, y: np.ndarray=None):
        self.removing_mask = np.sum(X, axis=0,  dtype=np.bool)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        # apply removing mask to every sample
        res = X[:, self.removing_mask]
        print(f"Shape after removing unused pixels: {res.shape[1]}")
        return res
