from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer

from lib.models import InputData
from local_models.preprocessing import CATEGORICAL_COLS, encode_categorical_cols


def run_inference(input_data: List[InputData], dv: DictVectorizer, model: BaseEstimator) -> np.ndarray:

    logger.info(f"Running inference on:\n{input_data}")
    df = pd.DataFrame([x.dict() for x in input_data])
    df = encode_categorical_cols(df)
    dicts = df[CATEGORICAL_COLS].to_dict(orient="records")
    X = dv.transform(dicts)
    y = model.predict(X)
    logger.info(f"Predicted wine quality:\n{y}")
    return y
