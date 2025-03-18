import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, CatBoostClassifier, XGBClassifier, LGBMClassifier,
    StackingClassifier, StackingRegressor
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score, silhouette_score
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError

class AutoMLAlgorithms:
    def __init__(self):
        self.models = {}

    # ------------------------ LINEAR & MULTI-LINEAR REGRESSION ------------------------ #
    def linear_regression(self, X, y, **params):
        default_params = {"fit_intercept": True}
        model_params = {**default_params, **params}

        try:
            model = LinearRegression(**model_params)
            model.fit(X, y)
            self.models['linear_regression'] = model

            predictions = model.predict(X)
            rmse = round(np.sqrt(mean_squared_error(y, predictions)), 4)
            r2 = round(r2_score(y, predictions), 4)

            print(f"✅ Linear Regression Trained Successfully! RMSE: {rmse}, R2 Score: {r2}")
            return model, {"RMSE": rmse, "R2_Score": r2}
        except Exception as e:
            raise RuntimeError(f"Linear Regression Error: {str(e)}")

    # ------------------------ POLYNOMIAL REGRESSION ------------------------ #
    def polynomial_regression(self, X, y, degree=2, **params):
        try:
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression(**params))
            model.fit(X, y)
            self.models['polynomial_regression'] = model

            predictions = model.predict(X)
            rmse = round(np.sqrt(mean_squared_error(y, predictions)), 4)
            r2 = round(r2_score(y, predictions), 4)

            print(f"✅ Polynomial Regression (Degree {degree}) Trained Successfully! RMSE: {rmse}, R2 Score: {r2}")
            return model, {"RMSE": rmse, "R2_Score": r2}
        except Exception as e:
            raise RuntimeError(f"Polynomial Regression Error: {str(e)}")

    # ------------------------ MEAN SHIFT ------------------------ #
    def mean_shift(self, X, **params):
        default_params = {"bandwidth": None}
        model_params = {**default_params, **params}

        try:
            model = MeanShift(**model_params)
            labels = model.fit_predict(X)
            self.models['mean_shift'] = model

            print(f"✅ Mean Shift Model Trained Successfully! Clusters Found: {len(set(labels))}")
            return model, {"Clusters Identified": len(set(labels))}
        except Exception as e:
            raise RuntimeError(f"Mean Shift Error: {str(e)}")

    # ------------------------ STACKING CLASSIFIER ------------------------ #
    def stacking_classifier(self, estimators, final_estimator, X, y, **params):
        try:
            model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, **params)
            model.fit(X, y)
            self.models['stacking_classifier'] = model

            accuracy = round(accuracy_score(y, model.predict(X)) * 100, 2)
            print(f"✅ Stacking Classifier Trained Successfully! Accuracy: {accuracy}%")
            return model, {"Accuracy (%)": accuracy}
        except Exception as e:
            raise RuntimeError(f"Stacking Classifier Error: {str(e)}")

    # ------------------------ STACKING REGRESSOR ------------------------ #
    def stacking_regressor(self, estimators, final_estimator, X, y, **params):
        try:
            model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, **params)
            model.fit(X, y)
            self.models['stacking_regressor'] = model

            predictions = model.predict(X)
            rmse = round(np.sqrt(mean_squared_error(y, predictions)), 4)
            r2 = round(r2_score(y, predictions), 4)

            print(f"✅ Stacking Regressor Trained Successfully! RMSE: {rmse}, R2 Score: {r2}")
            return model, {"RMSE": rmse, "R2_Score": r2}
        except Exception as e:
            raise RuntimeError(f"Stacking Regressor Error: {str(e)}")

    # ------------------------ PREDICTION FUNCTION ------------------------ #
    def predict(self, model_name, X_new):
        try:
            if model_name not in self.models:
                raise KeyError(f"❌ Error: Model '{model_name}' not found. Train the model first.")
            
            model = self.models[model_name]
            return model.predict(X_new)
        except NotFittedError:
            raise RuntimeError(f"❌ Error: Model '{model_name}' is not fitted yet.")
        except Exception as e:
            raise RuntimeError(f"❌ Prediction Error: {str(e)}")
