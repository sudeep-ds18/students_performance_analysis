import os
import joblib
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import joblib

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

from sklearn.metrics import r2_score
import sys
from src.exception import CustomException

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():

            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )
                gs.fit(x_train, y_train)

                best_model = gs.best_estimator_

            else:
                best_model = model
                best_model.fit(x_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)