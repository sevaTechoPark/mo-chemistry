import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import math

class FeaturesEngineeringVolumetricSurfaceMolecule(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transform_dict = {
            'EState_VSA': ['EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'EState_VSA11'],
            'VSA_EState': ['VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10'],
            'PEOE_VSA': ['PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14'],
            'SMR_VSA': ['SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SMR_VSA10'],
            'SlogP_VSA': ['SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12'],
        }
        for new_column, old_columns in transform_dict.items():
            X[new_column] = X.apply(lambda row: self.aggregate_axis_data(row, old_columns), axis=1)

        all_old_columns = []
        for values in transform_dict.values():
            all_old_columns.extend(values)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def aggregate_axis_data(self, row, columns):
        sum_of_squares = sum(row[column]**2 for column in columns)
        return math.sqrt(sum_of_squares)

class FeaturesEngineeringDensityMorganFingerprints(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transform_dict = {
            'FpDensityMorgan': ['FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3'],
        }
        for new_column, old_columns in transform_dict.items():
            X[new_column] = X.apply(lambda row: self.aggregate_axis_data(row, old_columns), axis=1)

        all_old_columns = []
        for values in transform_dict.values():
            all_old_columns.extend(values)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def aggregate_axis_data(self, row, columns):
        sum_of_squares = sum(row[column]**2 for column in columns)
        return sum_of_squares

class FeaturesEngineeringChiIndices(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        columns = [
            'Chi0', 'Chi0n',
            'Chi1n', 'Chi1v', 
            'Chi2n', 'Chi2v',
            'Chi3n', 'Chi3v',
            'Chi4n', 'Chi4v',
        ]
        X['Chi_sum'] = X[columns].sum(axis=1)
        X['Chi_std'] = X[columns].std(axis=1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class FeaturesEngineeringKappa(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transform_dict = {
            'Kappa': ['Kappa1', 'Kappa2', 'Kappa3'],
        }
        for new_column, old_columns in transform_dict.items():
            X[new_column] = X.apply(lambda row: self.aggregate_axis_data(row, old_columns), axis=1)

        all_old_columns = []
        for values in transform_dict.values():
            all_old_columns.extend(values)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def aggregate_axis_data(self, row, columns):
        sum_of_squares = sum(row[column]**2 for column in columns)
        return sum_of_squares

class FeaturesEngineeringBCUT(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        columns = [
            'BCUT2D_MWHI',
            'BCUT2D_MWLOW',
            'BCUT2D_CHGHI',
            'BCUT2D_CHGLO',
            'BCUT2D_LOGPHI',
            'BCUT2D_LOGPLOW',
            'BCUT2D_MRHI',
            'BCUT2D_MRLOW',
        ]
        X['BCUT2D_sum'] = X[columns].sum(axis=1)
        X['BCUT2D_std'] = X[columns].std(axis=1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class FeaturesEngineeringComplexScore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Преобразует данные для создания нового признака `complex_score`.
    
        MaxAbsEStateIndex и qed используются для оценки электронной структуры
        и "лекарственности" молекулы. Их произведение, дополненное
        экспоненциальным преобразованием qed, подчеркивает влияние
        "лекарственности".
    
        SPS добавляется в знаменатель, чтобы отразить фактор сложности синтеза; добавление 1 в знаменатель предотвращает деление на ноль
    
        Таким образом, новый признак complex_score складывает вместе влияние всех аспектов молекулы на её свойства, и этот подход может улучшить выявление скрытых взаимодействий.
        """

        X['complex_score'] = (X['MaxAbsEStateIndex'] * np.exp(df['qed'])) / (df['SPS'] + 1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MaxAbsScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

def create_preprocessing_pipeline():
    return Pipeline([
        ('scaler', DataFrameScaler()),
        ('features_engineering_volumetric', FeaturesEngineeringVolumetricSurfaceMolecule()),
        ('features_engineering_density', FeaturesEngineeringDensityMorganFingerprints()),
        ('features_engineering_chi', FeaturesEngineeringChiIndices()),
        ('features_kappa', FeaturesEngineeringKappa()),
        ('features_bcut', FeaturesEngineeringBCUT()),
    ])