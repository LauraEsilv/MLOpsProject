from typing import List
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

CATEGORICAL_COLS = ['type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
       'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
       'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']


def encode_categorical_cols(wine_df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ['type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
                             'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                             'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    le = preprocessing.LabelEncoder()
    wine_df['type'] = le.fit_transform(wine_df['type'])

    # Ensure that the numeric columns are present before attempting to fill missing values
    numeric_cols = [col for col in categorical_cols if col in wine_df.columns]
    
    # Fill missing values for numeric columns
    wine_df[numeric_cols] = wine_df[numeric_cols].fillna(0).astype("float")

    return wine_df


def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> dict:

    if categorical_cols is None:
        categorical_cols = ['type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
                             'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                             'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    dicts = df[categorical_cols].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["quality"].values

    x = dv.transform(dicts)
    return x, y, dv