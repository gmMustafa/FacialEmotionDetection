#! .venv\Scripts\python.exe

import pandas as pd
import numpy as np

def mean_std_of_data(csvdatafile):
    df = pd.read_csv(csvdatafile)
    # Extract features (X) and labels (y)
    X = df.drop("Category", axis=1)
    X_mean = X.mean()
    X_std = X.std()
    X_mean_array = np.array(X_mean).reshape(1, -1)
    X_std_array = np.array(X_std).reshape(1, -1)
    return X_mean_array, X_std_array