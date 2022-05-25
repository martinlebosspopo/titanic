from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, *args):
        return args[0]


class ColumnsAsParameters(BaseTransformer):
    def __init__(self, cols_ignore=[], cols_select=[]):
        self.cols_ignore = cols_ignore
        self.cols_select = cols_select
        self.cols_final = None

    def fit(self, X, y=None):
        self.cols_final = self.cols_select or [col for col in X.columns if
                                               col not in self.cols_ignore]
        self.cols_ignore = [col for col in X.columns if col not in self.cols_final]
        return self

    def split_ignored_final(self, X):
        return X[self.cols_ignore], X[self.cols_final]

    def format_df(self, X, X_transformed_arr):
        df_tr = pd.DataFrame(X_transformed_arr, columns=self.cols_final, index=X.index)
        return pd.concat([X[self.cols_ignore], df_tr], axis=1)


class PandasTransformer(ColumnsAsParameters):
    def __init__(self, cols_ignore=[], cols_select=[]):
        super().__init__(cols_select=cols_select, cols_ignore=cols_ignore)
        self.skmodel = None

    def fit(self, X, y=None):
        super().fit(X)
        self.skmodel.fit(X[self.cols_final])
        return self

    def transform(self, X, y=None):
        X_ignored, X_cols_final = self.split_ignored_final(X)
        X_transformed_arr = self.skmodel.transform(X_cols_final)
        return self.format_df(X, X_transformed_arr)
