import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from tqdm import tqdm

class FeatureSelector:
    def __init__(self, model, start_index=5, step=5):
        self.model = model
        self.start_index = start_index
        self.step = step

        self.kbest_selected_features_list = []
        self.kbest_mae_scores = []
        self.kbest_r2_scores = []
        self.kbest_features_index = None

        self.rfe_selected_features_list = []
        self.rfe_mae_scores = []
        self.rfe_r2_scores = []
        self.rfe_features_index = None

    def calculate_k_best(self, X_train, X_test, y_train, y_test):
        for n_features in tqdm(range(self.start_index, X_train.shape[1] + 1, self.step), desc="K-Best Progress"):
            selector = SelectKBest(score_func=f_regression, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            self.model.fit(X_train_selected, y_train)
            y_pred = self.model.predict(X_test_selected)

            selected_features = X_train.columns[selector.get_support()]
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.kbest_selected_features_list.append(selected_features)
            self.kbest_mae_scores.append(mae)
            self.kbest_r2_scores.append(r2)

    def calculate_rfe(self, X_train, X_test, y_train, y_test):
        for n_features in tqdm(range(self.start_index, X_train.shape[1] + 1, self.step), desc="RFE Progress"):
            rfe = RFE(estimator=self.model, n_features_to_select=n_features)
            X_train_selected = rfe.fit_transform(X_train, y_train)
            X_test_selected = rfe.transform(X_test)

            self.model.fit(X_train_selected, y_train)
            y_pred = self.model.predict(X_test_selected)

            selected_features = X_train.columns[rfe.support_]
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.rfe_selected_features_list.append(selected_features)
            self.rfe_mae_scores.append(mae)
            self.rfe_r2_scores.append(r2)

    def select_kbest_features(self, count):
        self.kbest_features_index = int(count / self.step) - 1

    def print_kbest_result(self):
        kbest_selected_features = self.kbest_selected_features_list[self.kbest_features_index]
        print(f'mae: {self.kbest_mae_scores[self.kbest_features_index]}')
        print(f'r2: {self.kbest_r2_scores[self.kbest_features_index]}')
        print(kbest_selected_features)

    def select_rfe_features(self, count):
        self.rfe_features_index = int(count / self.step) - 1

    def print_rfe_result(self):
        rfe_selected_features = self.rfe_selected_features_list[self.rfe_features_index]
        print(f'mae: {self.rfe_mae_scores[self.rfe_features_index]}')
        print(f'r2: {self.rfe_r2_scores[self.rfe_features_index]}')
        print(rfe_selected_features)

    def get_selected_features(self):
        kbest_selected_features = self.kbest_selected_features_list[self.kbest_features_index]
        rfe_selected_features = self.rfe_selected_features_list[self.rfe_features_index]
        
        common_selected_features = kbest_selected_features.intersection(rfe_selected_features)
        combined_selected_features = list(set(kbest_selected_features) | set(rfe_selected_features))
        return common_selected_features, combined_selected_features
    
    def draw_selection(self, scores, score_name, target_name, filter_method_name):
        plt.figure(figsize=(14, 8))

        plt.plot(range(self.start_index, (len(scores) + 1) * self.step, self.step), scores, marker='o')
        
        plt.title(f'Зависимость метрики {score_name} таргета "{target_name}" от количества признаков {filter_method_name}')
        plt.xlabel('Количество признаков')
        plt.ylabel(score_name)

        plt.xticks(range(self.step * 2, (len(scores) + 1) * self.step, self.step * 2))
        for x in range(self.step * 2, (len(scores) + 1) * self.step, self.step * 2):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)

        plt.show()
