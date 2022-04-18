from .base import BaseTransformer, ColumnsAsParameters

import sklearn.compose as com
import sklearn.impute as imp
import sklearn.preprocessing as pre

import pandas as pd


class ColumnTransformer(BaseTransformer):
    """ Column Transformer for pandas DFs"""

    def __init__(self, transformers, remainder='passthrough', **kwargs):
        self.col_transformer = com.ColumnTransformer(transformers, remainder=remainder)
        self.col_transformer.set_params(**kwargs)
        self.transformers = transformers
        self.remainder = remainder
        self.cols = None

    def fit(self, X, y=None):
        self.col_transformer.fit(X)
        self.cols = [col.split('__')[1] for col in self.col_transformer.get_feature_names_out()]
        return self

    def get_params(self, deep=True):
        params_this = super().get_params()
        params_colTran = self.col_transformer.get_params()
        params_colTran_deep = {key:val for key, val in params_colTran.items() if '__' in key}
        params_this.update(params_colTran_deep)
        return params_this

    def set_params(self, **kwargs):
        params = {key: val for key, val in kwargs.items() if '__' not in key}
        sub_params = {key: val for key, val in kwargs.items() if '__' in key}
        self.col_transformer.set_params(**sub_params)
        super().set_params(**params)

    def transform(self, X, y=None):
        arr_transfo = self.col_transformer.transform(X)
        return pd.DataFrame(arr_transfo, columns=self.cols, index=X.index)


class OneHotEncoder(ColumnsAsParameters):
    def __init__(self, cols_ignore=[], cols_select=[]):
        super().__init__(cols_select=cols_select, cols_ignore=cols_ignore)
        self.one_encoder = pre.OneHotEncoder(sparse=False)

    def fit(self, X, y=None):
        super().fit(X, y=y)
        self.one_encoder.fit(X[self.cols_final])
        return self

    def transform(self, X, y=None):
        arr_one = self.one_encoder.transform(X[self.cols_final])
        df_one = pd.DataFrame(
            arr_one,
            columns=self.one_encoder.get_feature_names_out(),
            index=X.index
        )
        return pd.concat([X.drop(self.cols_final, axis=1), df_one], axis=1)


class SimpleImputer(BaseTransformer):
    """ SimpleImputer for pandas DFs"""

    def __init__(self, strategy):
        self.imputer = imp.SimpleImputer(strategy=strategy)
        self.strategy = strategy
        self.cols = None

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.cols = X.columns
        return self

    def get_feature_names_out(self, *args, **kwargs):
        return self.cols

    def transform(self, X, y=None):
        transfo_arr = self.imputer.transform(X)
        return pd.DataFrame(transfo_arr, columns=self.cols, index=X.index)


class StandardScaler(BaseTransformer):
    """ StandardScaler for pandas DFs"""

    def __init__(self):
        self.scaler = pre.StandardScaler()
        self.cols = None

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.cols = X.columns
        return self

    def get_feature_names_out(self, *args, **kwargs):
        return self.cols

    def transform(self, X, y=None):
        transfo_arr = self.scaler.transform(X)
        return pd.DataFrame(transfo_arr, columns=self.cols, index=X.index)