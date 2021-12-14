import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def evaluate(model, X, y):
    preds = model.predict(X)
    print("RMSLE: " + str(rmsle(preds, y)))
    
    
    
def to_categorical(X):
    for c in X.columns:
        col_type = X[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            X[c] = X[c].astype('category')

def submission(transformer, gs_model, create_sub_file: bool=False, obj_to_num: bool=False):
    cheat_path = os.path.join("data", "result-with-best.csv")
    cheat = pd.read_csv(cheat_path)
    
    test_path = os.path.join("data", "test.csv")
    validation = pd.read_csv(test_path)
    val_ids = validation["Id"]
    validation = validation.drop(columns=["Id"])

    validation = transformer.fit_transform(validation, obj_to_num=obj_to_num)
    to_categorical(validation)

    sub_predictions = gs_model.predict(validation)
    print("RMSLE submission: " + str(rmsle(sub_predictions, np.log1p(cheat["SalePrice"]))))
    
    if create_sub_file:
        d = {'Id': val_ids.to_numpy(), 'SalePrice':  np.expm1(sub_predictions)}
        df = pd.DataFrame(data=d)
        df.to_csv('submission.csv', index=False)
        

def load_data():
    train_path = os.path.join("data", "train.csv")
    train_df: pd.DataFrame = pd.read_csv(train_path).drop("Id", axis=1)

    target = np.log1p(train_df['SalePrice'])
    train_df: pd.DataFrame = train_df.drop(columns='SalePrice', axis=1)
    return train_df, target
    