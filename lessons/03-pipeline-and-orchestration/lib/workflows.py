# To complete
import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model_workflow(
    train_filepath: str,
    test_filepath: str,
    artifacts_filepath: Optional[str] = None,
) -> dict:
    logger.info("Starting the training process...")
    
    # Load and process training data
    X_train, y_train, _ = process_data(train_filepath)

    # Train the model
    logger.info("Training the model...")
    model = train_model(X_train, y_train)

    # Evaluate the model
    logger.info("Evaluating the model...")
    X_test, y_test, _ = process_data(test_filepath, dv=model.dv, with_target=True)
    y_pred = predict(X_test, model)
    mse = evaluate_model(y_test, y_pred)
    
    logger.info("Training process completed.")

def batch_predict_workflow(
    input_filepath: str,
    model: Optional[LinearRegression] = None,
    dv: Optional[DictVectorizer] = None,
    artifacts_filepath: Optional[str] = None,
) -> np.ndarray:
    """Perform the whole prediction process."""
    logger.info("Starting the batch prediction process...")

    # Process input data
    X, _, dv = process_data(input_filepath, dv=dv, with_target=False)
    
    # Predict using the provided model or load the model from artifacts
    if model is None and artifacts_filepath:
        logger.info(f"Loading model from {artifacts_filepath}...")
        with open(artifacts_filepath, 'rb') as f:
            model = np.load(f)
    
    if model is None:
        raise ValueError("Model is required for prediction.")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = predict(X, model)
    
    logger.info("Batch prediction process completed.")
    
    return predictions