import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def eda(df):
    df = df.copy()

    # computing age of vehicle
    year_mode = df[df.YearMade > 1000].YearMade.mode()
    df['YearMade'] = df.YearMade.replace(1000, year_mode[0])
    df['saledate'] = pd.to_datetime(df.saledate)
    df['age'] = df.saledate.dt.year - df.YearMade

    # one-hot encoding product group
    pgdf = pd.get_dummies(df.ProductGroup)
    df[pgdf.columns] = pgdf

    # one-hot encoding enclosure
    encdf = pd.get_dummies(df.Enclosure.fillna(df.Enclosure.mode()))
    df[encdf.columns] = encdf

    # final features
    columns = ['age', 'BL', 'MG', 'SSL', 'TEX', 'TTT', 'OROPS', 'EROPS', 'EROPS w AC']
    return df[columns]

def cross_validation(model, X, y, k, metrics):
    matrix = np.empty((k, len(metrics)))
    for i, (train_index, val_index) in enumerate(KFold(n_splits=k).split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = model.fit(X_train, y_train)
        y_hat = model.predict(X_val)
        for j, metric in enumerate(metrics):
            matrix[i, j] = metric(y_val, y_hat)
    return matrix.mean(axis=0)

def rmse(actual, predicted):
    return mean_squared_error(actual, predicted) ** 0.5

def rmsle(actual, predicted):
    return ((np.log(predicted + 1) - np.log(actual + 1)) ** 2).mean() ** 0.5
