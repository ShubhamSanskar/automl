import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, RFE, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class AutoMLAdvancedFeatures:
    def __init__(self):
        self.models = {}

    # ------------------------ ERROR ANALYSIS ------------------------ #
    def error_analysis(self, model, X, y, model_name="Model"):
        try:
            predictions = model.predict(X)
            residuals = y - predictions

            plt.figure(figsize=(12, 5))

            # Residual Plot
            plt.subplot(1, 2, 1)
            sns.residplot(x=predictions, y=residuals, lowess=True, line_kws={'color': 'red'})
            plt.title(f"{model_name} - Residual Plot")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")

            # Error Distribution
            plt.subplot(1, 2, 2)
            sns.histplot(residuals, kde=True, bins=30)
            plt.title(f"{model_name} - Error Distribution")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error Analysis Error: {str(e)}")

    # ------------------------ FEATURE SELECTION - SELECTKBEST ------------------------ #
    def feature_selection_kbest(self, X, y, k=5):
        try:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_new = selector.fit_transform(X, y)
            
            selected_features = list(X.columns[selector.get_support()])
            print(f"✅ SelectKBest Feature Selection Completed! Selected Features: {selected_features}")
            return X_new, selected_features
        except Exception as e:
            raise RuntimeError(f"SelectKBest Error: {str(e)}")

    # ------------------------ FEATURE SELECTION - RFE ------------------------ #
    def feature_selection_rfe(self, model, X, y, n_features_to_select=5):
        try:
            selector = RFE(model, n_features_to_select=n_features_to_select)
            X_new = selector.fit_transform(X, y)
            
            selected_features = list(X.columns[selector.get_support()])
            print(f"✅ RFE Feature Selection Completed! Selected Features: {selected_features}")
            return X_new, selected_features
        except Exception as e:
            raise RuntimeError(f"RFE Error: {str(e)}")

    # ------------------------ HYPERPARAMETER TUNING - GRID SEARCH ------------------------ #
    def grid_search_tuning(self, model, X, y, param_grid):
        try:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X, y)
            
            print(f"✅ GridSearchCV Best Parameters Found: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            raise RuntimeError(f"GridSearchCV Error: {str(e)}")

    # ------------------------ HYPERPARAMETER TUNING - RANDOMIZED SEARCH ------------------------ #
    def randomized_search_tuning(self, model, X, y, param_distributions, n_iter=10):
        try:
            random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=5, scoring='accuracy', random_state=42)
            random_search.fit(X, y)

            print(f"✅ RandomizedSearchCV Best Parameters Found: {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            raise RuntimeError(f"RandomizedSearchCV Error: {str(e)}")
