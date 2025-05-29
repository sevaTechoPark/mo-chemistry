import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, Normalizer, RobustScaler, PowerTransformer
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
            'SMR_VSA': ['SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10'],
            'SlogP_VSA': ['SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12'],
        }
        for new_column, old_columns in transform_dict.items():
            X[new_column + '_sum'] = X.apply(lambda row: row[old_columns].sum(), axis=1)
            X[new_column + '_mean'] = X.apply(lambda row: row[old_columns].mean(), axis=1)
            X[new_column + '_median'] = X.apply(lambda row: row[old_columns].median(), axis=1)
            X[new_column + '_max'] = X.apply(lambda row: row[old_columns].max(), axis=1)
            X[new_column + '_min'] = X.apply(lambda row: row[old_columns].min(), axis=1)
            X[new_column + '_std'] = X.apply(lambda row: row[old_columns].std(), axis=1)
            X[new_column + '_sqrt'] = X.apply(lambda row: ((row[old_columns]**2).sum()**(1/2)), axis=1)
            X[new_column + '_prod'] = X.apply(lambda row: row[old_columns].prod(), axis=1)
            X[new_column + '_variation'] = X[new_column + '_std'] / X[new_column + '_mean']

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class FeaturesEngineeringChiIndices(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        columns = [
            'Chi0', 'Chi0n', 'Chi0v',
            'Chi1', 'Chi1n', 'Chi1v', 
            'Chi2n', 'Chi2v',
            'Chi3n', 'Chi3v',
            'Chi4n', 'Chi4v',
        ]
        X['Chi_sum'] = X[columns].sum(axis=1)
        X['Chi_mean'] = X[columns].mean(axis=1)
        X['Chi_median'] = X[columns].median(axis=1)
        X['Chi_max'] = X[columns].max(axis=1)
        X['Chi_min'] = X[columns].min(axis=1)
        X['Chi_std'] = X[columns].std(axis=1)
        X['Chi_sqrt'] = X.apply(lambda row: ((row[columns]**2).sum()**(1/2)), axis=1)
        X['Chi_prod'] = X[columns].prod(axis=1)
        X['Chi_variation'] = X['Chi_std'] / X['Chi_mean']

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

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
        X['BCUT2D_mean'] = X[columns].mean(axis=1)
        X['BCUT2D_median'] = X[columns].median(axis=1)
        X['BCUT2D_max'] = X[columns].max(axis=1)
        X['BCUT2D_min'] = X[columns].min(axis=1)
        X['BCUT2D_std'] = X[columns].std(axis=1)
        X['BCUT2D_sqrt'] = X.apply(lambda row: ((row[columns]**2).sum()**(1/2)), axis=1)
        X['BCUT2D_prod'] = X[columns].prod(axis=1)
        X['BCUT2D_variation'] = X['BCUT2D_std'] / X['BCUT2D_mean']

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class FeaturesEngineeringDensityMorganFingerprints(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transform_dict = {
            'FpDensityMorgan': ['FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3'],
        }
        for new_column, old_columns in transform_dict.items():
            X[new_column] = X.apply(lambda row: self.aggregate_axis_data(row, old_columns), axis=1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def aggregate_axis_data(self, row, columns):
        sum_of_squares = sum(row[column]**2 for column in columns)
        return sum_of_squares

class FeaturesEngineeringKappa(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transform_dict = {
            'Kappa': ['Kappa1', 'Kappa2', 'Kappa3'],
        }
        for new_column, old_columns in transform_dict.items():
            X[new_column] = X.apply(lambda row: self.aggregate_axis_data(row, old_columns), axis=1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def aggregate_axis_data(self, row, columns):
        sum_of_squares = sum(row[column]**2 for column in columns)
        return sum_of_squares

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

        X['complex_score'] = (X['MaxAbsEStateIndex'] * np.exp(X['qed'])) / (X['SPS'] + 1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        # MinMaxScaler не сильно портит, но портит
        # Normalizer немного портит ic, повышает немного cc, и сильно растит si
        # PowerTransformer(method='yeo-johnson') не сильно портят
        self.scaler = Normalizer()

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
        ('features_engineering_chi', FeaturesEngineeringChiIndices()),
        ('features_bcut', FeaturesEngineeringBCUT()),
        ('features_engineering_density', FeaturesEngineeringDensityMorganFingerprints()),
        ('features_kappa', FeaturesEngineeringKappa()),
        ('features_complex_score', FeaturesEngineeringComplexScore()),
    ])
