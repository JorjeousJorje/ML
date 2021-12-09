import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from DataProcessingUtils import *

class DataTransformer:
    
    
    def __init__(self, scaler=None):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

        self.imputer = IterativeImputer(max_iter=10,
                                        random_state=0)
    
    def prepare(self, X):
        X = delete_nans(X, 0.45)
        X = fill_nans(X)
        return X
    
    def fit_transform(self, X):
        
        X = self.prepare(X)
        num_candidates = list(X.dtypes[X.dtypes != "object"].index.values)
        X[num_candidates] = self.imputer.fit_transform(X[num_candidates])
        X[num_candidates] = self.scaler.fit_transform(X[num_candidates])   
        return X     
    