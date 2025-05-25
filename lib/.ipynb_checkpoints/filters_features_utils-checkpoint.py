import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE

def selectKBest(model, X_train, X_test, y_train, y_test):
    selected_features_list = []
    mae_scores = []
    r2_scores = []
    
    for n_features in range(5, X_train.shape[1] + 1):
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        selected_features = X_train.columns[selector.get_support()]
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        selected_features_list.append(selected_features)
        mae_scores.append(mae)
        r2_scores.append(r2)
    return (mae_scores, r2_scores, selected_features_list)    

def selectRFE(model, X_train, X_test, y_train, y_test):
    selected_features_list = []
    mae_scores = []
    r2_scores = []
    
    for n_features in range(5, X_train.shape[1] + 1):
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        X_train_selected = rfe.fit_transform(X_train, y_train)
        X_test_selected = rfe.transform(X_test)
        
        model.fit(X_train_selected, y_train)
        
        y_pred = model.predict(X_test_selected)

        selected_features = X_train.columns[rfe.support_]
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        selected_features_list.append(selected_features)
        mae_scores.append(mae)
        r2_scores.append(r2)
    return (mae_scores, r2_scores, selected_features_list)    

def drawSelection(scores, score_name, target_name, filter_method):
    plt.figure(figsize=(14, 8))
    plt.plot(range(1, len(scores) + 1), scores, marker='o')
    plt.title(f'Зависимость {score_name} {target_name} от количества выбранных признаков')
    plt.xlabel('Количество выбранных признаков')
    plt.ylabel(f'{score_name} {filter_method}')
    plt.xticks(range(10, len(scores) + 1, 10))
    for x in range(10, len(scores) + 1, 10):
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)
    plt.show()
