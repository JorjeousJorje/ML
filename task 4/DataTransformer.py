import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy.stats as stats



class DataTransformer:
    
    def __init__(self, scaler=None):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

        self.encoder = LabelEncoder()
        self.imputer = IterativeImputer(max_iter=10, random_state=0)
        
    def fillnans(self, X: pd.DataFrame):
        pd.options.mode.chained_assignment = None
        
        # fill with mode
        X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])
        X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])
        X['MSZoning'] = X['MSZoning'].fillna(X['MSZoning'].mode()[0])
        
        dist_cols = ["LotFrontage", "GarageYrBlt"]
        
        for col in dist_cols:
            missing = X[X[col].isna()][col]
            not_missing = X[X[col].notnull()][col]

            params = stats.johnsonsu.fit(not_missing)

            r = stats.johnsonsu.rvs(params[0], params[1], params[2], params[3], size=missing.shape[0])
            X[col].loc[missing.index] = r

            
        zero_nan_cols = ['GarageArea', 'GarageCars', "MasVnrArea", 'BsmtFinSF1', 'BsmtFinSF2',
                            'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr',
                            "1stFlrSF", "2ndFlrSF"]
        
        X[zero_nan_cols] = X[zero_nan_cols].fillna(0)
        
        X['Functional'] = X['Functional'].fillna('Typ')  # Typical Functionality
        X['Electrical'] = X['Electrical'].fillna("SBrkr")  # Standard Circuit Breakers & Romex
        X['KitchenQual'] = X['KitchenQual'].fillna("TA")  # Typical/Average
        
        rest_cols = X.select_dtypes("object").fillna("None").astype("object")
        X[rest_cols.columns] = rest_cols.values
        
        return X
            
        
    def prepare(self, X):
        X = self.fillnans(X)
        return X
    
    def encode(self, X):
        object_candidates = list(X.dtypes[X.dtypes == "object"].index.values)
        
        for col in object_candidates:
            X[col] = self.encoder.fit_transform(X[col])
        
        return X
    
    def fit_transform(self, X, obj_to_num: bool=False):
        
        X = self.prepare(X)
        num_candidates = list(X.dtypes[X.dtypes != "object"].index.values)
        X[num_candidates] = self.imputer.fit_transform(X[num_candidates])
        
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['Total_sqr_footage'] = (X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF'])
        X['Total_porch_sf'] = (X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF'])
        
        if obj_to_num:
            X = self.encode(X)
            
        return X     
    