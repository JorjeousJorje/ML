import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class UnusedPixelsRemover(BaseEstimator, TransformerMixin):
    
    def get_removing_mask(self, X: np.ndarray) -> np.ndarray:
        mask: np.ndarray = np.zeros_like(X[0])
    
        for sample in X:
            mask = (mask != sample)
        
        return mask
    
    def __init__(self) -> None:
        self.removing_mask: np.ndarray = None
    
    
    def fit(self, X: np.ndarray, y: np.ndarray=None):
        self.removing_mask = self.get_removing_mask(X)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        # apply removing mask to every sample
        res = X[:, self.removing_mask]
        print(f"Shape after removing unused pixels: {res.shape[1]}")
        return res
