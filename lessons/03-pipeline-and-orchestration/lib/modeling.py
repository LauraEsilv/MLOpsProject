# To complete
import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

@task(name="Train_model", tags=["Serialize"])
def train_model(X: scipy.sparse.csr_matrix, y: np.ndarray) -> LinearRegression:
    lr = LinearRegression()
    lr.fit(X, y)
    return lr

@task(name="Predict", tags=["Serialize"])
def predict(X: scipy.sparse.csr_matrix, model: LinearRegression) -> np.ndarray:
    return model.predict(input_data)

@task(name="Evaluate_model", tags=["Serialize"])
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)