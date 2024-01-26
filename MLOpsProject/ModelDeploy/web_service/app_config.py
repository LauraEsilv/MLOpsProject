# MODELS
MODEL_VERSION = "0.0.1"
PATH_TO_PREPROCESSOR = "local_models/preprocessing.py"
PATH_TO_MODEL = f"local_models/model.pkl"
CATEGORICAL_VARS = ['wine_type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
       'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
       'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']


# MISC
APP_TITLE = "WineQualityPrediction"
APP_DESCRIPTION = (
    "A simple API to predict wine quality "
    "dependng on different features "
)
APP_VERSION = "0.0.1"
