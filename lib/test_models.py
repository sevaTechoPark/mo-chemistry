import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor

def run_models_regressions(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'KNeighbors': KNeighborsRegressor(2),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='mae', random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1, max_depth=5, random_state=42),
        'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42),
        'AdaBoostRegressor': AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, learning_rate=1.0, random_state=42),
        'Krr': KernelRidge(alpha=0.1, kernel='rbf', gamma=0.1),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if r2 > 0:
            mae = round(mae, 2)
            r2 = round(r2, 2)
            results.append({'Model': name, 'Mean Absolute Error': mae, 'R2 Score': r2})

    return pd.DataFrame(results)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier

def run_models_classifications(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNeighbors': KNeighborsClassifier(2),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, verbose=-1, max_depth=5, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0),
        'HistGradientBoostingRegressor': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42),
        'AdaBoostRegressor': AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=3), n_estimators=50, learning_rate=1.0, algorithm='SAMME', random_state=42),
        # 'SVC': SVC(kernel='rbf'),
    }

    result_df = pd.DataFrame()
    roc_curves = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        if accuracy > 0:
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_score)

            report_df = pd.DataFrame(report_dict).transpose()
            report_df.loc['accuracy'] = [accuracy, np.nan, np.nan, np.nan]
            report_df.loc['roc_auc'] = [roc_auc, np.nan, np.nan, np.nan]
            report_df['Model'] = name

            result_df = pd.concat([result_df, report_df], axis=0)


    return result_df
    