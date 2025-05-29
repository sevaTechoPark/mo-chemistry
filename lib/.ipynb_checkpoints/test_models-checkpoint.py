import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

def run_models_regressions(X_train, X_test, y_train, y_test, log_target=False):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'KNeighbors': KNeighborsRegressor(2),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='mae', random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1, max_depth=5, random_state=42),
        'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0),
        'Krr': KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
    }

    results = []
    for name, model in models.items():
        if log_target:
            y_train_log = np.log1p(y_train)
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'Mean Absolute Error': mae, 'R2 Score': r2})

    return pd.DataFrame(results)
