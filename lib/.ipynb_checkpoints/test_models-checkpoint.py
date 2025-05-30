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
from sklearn.naive_bayes import GaussianNB

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
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42),
        'AdaBoost': AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, learning_rate=1.0, random_state=42),
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

    return pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)


from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
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
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42),
        'AdaBoost': AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=3), n_estimators=50, learning_rate=1.0, algorithm='SAMME', random_state=42),
        'Gaussian': GaussianNB(),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_score)

        precision = report_dict['weighted avg']['precision']
        recall = report_dict['weighted avg']['recall']
        f1_score = report_dict['weighted avg']['f1-score']
        support = report_dict['weighted avg']['support']

        accuracy = round(accuracy, 2)
        roc_auc = round(roc_auc, 2)
        precision = round(precision, 2)
        recall = round(recall, 2)
        f1_score = round(f1_score, 2)
        support = round(support, 2)

        if f1_score > 0.5:
            results.append({
                'Model': name,
                'WA f1': f1_score,
                'accuracy': accuracy,
                'WA precision': precision,
                'WA recall': recall,
                'WA support': support,
                'roc_auc': roc_auc,
            })

    return pd.DataFrame(results).sort_values(by='WA f1', ascending=False)
    