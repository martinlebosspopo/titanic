import numpy as np
import pandas as pd
import sklearn.compose as com
import sklearn.impute as imp
import sklearn.preprocessing as pre

from .base import BaseTransformer, ColumnsAsParameters, PandasTransformer


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
        params_colTran_deep = {key: val for key, val in params_colTran.items() if '__' in key}
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
    IMPOSSIBLE_VALUE = 'WXMOI1029847DSO892Y3HFD12O3O8F7DQ87F2173797CUHWLD710287FSDF'

    def __init__(self, cols_ignore=[], cols_select=[], keep_nans=False, min_frequency=10):
        super().__init__(cols_select=cols_select, cols_ignore=cols_ignore)
        self.keep_nans = keep_nans
        self.min_frequency = min_frequency
        if keep_nans:
            assert self.min_frequency >= 2, "min_frequency must be >= 2"
            self.one_encoder = pre.OneHotEncoder(
                sparse=False,
                handle_unknown='infrequent_if_exist',
                min_frequency=self.min_frequency
            )
            self.cols_infrequent = []
        else:
            self.one_encoder = pre.OneHotEncoder(sparse=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        super().fit(X)
        # Add a row with an impossible value for all cols so that infrequent_sklearn will be
        # built for all columns
        X_fit = X[self.cols_final].dropna().astype(str)
        if self.keep_nans:
            imp_row = [{col: self.IMPOSSIBLE_VALUE for col in self.cols_final}]
            X_fit = X_fit.append(imp_row)

        self.one_encoder.fit(X_fit)
        self.cols_infrequent = [col for col in self.one_encoder.get_feature_names_out()
                                if col.endswith('_infrequent_sklearn')]
        return self

    def transform(self, X, y=None):
        X_ignored, X_cols_final = self.split_ignored_final(X)

        null_cells = X_cols_final.isna()
        X_cols_final = X_cols_final.astype(str).mask(null_cells, np.NaN)

        arr_one = self.one_encoder.transform(X_cols_final)
        df_one = pd.DataFrame(
            arr_one,
            columns=self.one_encoder.get_feature_names_out(),
            index=X.index
        )
        if self.keep_nans:
            for col_inft in self.cols_infrequent:
                col_basename = col_inft.split('_infrequent_sklearn')[0]
                col_cats = [col for col in df_one.columns if col.startswith(col_basename)]
                df_one[col_cats] = df_one[col_cats].where(df_one[col_inft] == 0)
            df_one = df_one.drop(self.cols_infrequent, axis=1)

        return pd.concat([X_ignored, df_one], axis=1)

    def inverse_transform(self, X):
        cols_features_out = [col for col in self.one_encoder.get_feature_names_out()
                             if col not in self.cols_infrequent]
        X_ignored = X.drop(cols_features_out, axis=1)
        X_cols_features_out = X[cols_features_out]
        X_cols_features_out[self.cols_infrequent] = 0  # Can be empty
        X_cols_features_out = X_cols_features_out[self.one_encoder.get_feature_names_out()]
        X_inversed_arr = self.one_encoder.inverse_transform(X_cols_features_out)
        df = pd.DataFrame(
            X_inversed_arr,
            columns=self.cols_final,
            index=X.index
        )
        return pd.concat([X_ignored, df], axis=1)

    def get_grouped_features(self):
        if not self.one_encoder:
            raise Exception('Encoder not fit')
        features = self.one_encoder.feature_names_in_
        categories = self.one_encoder.categories_
        inf_cat = self.one_encoder.infrequent_categories_
        res = []
        for feat, cats, infcat in zip(features, categories, inf_cat):
            cats = [c for c in cats if c not in infcat]
            res.append([(feat + '_' + c) for c in cats])
        return res


class KNNImputer(ColumnsAsParameters):
    """ SimpleImputer for pandas DFs"""

    def __init__(self, cols_ignore=[], cols_select=[]):
        super().__init__(cols_select=cols_select, cols_ignore=cols_ignore)
        self.imputer = imp.KNNImputer()
        self.group_features = []

    def fit(self, X, y=None):
        super().fit(X)
        self.imputer.fit(X[self.cols_final])
        return self

    def transform(self, X, y=None):
        X_ignored, X_cols_final = self.split_ignored_final(X)

        X_transformed_arr = self.imputer.transform(X_cols_final)
        df_tr = pd.DataFrame(X_transformed_arr, columns=self.cols_final, index=X.index)
        for col_cats in self.group_features:
            maxs = df_tr[col_cats].max(axis=1)
            for col in col_cats:
                df_tr[col] = (df_tr[col] == maxs).astype(float)
        return pd.concat([X_ignored, df_tr], axis=1)


class SimpleImputer(PandasTransformer):
    """ SimpleImputer for pandas DFs"""

    def __init__(self, strategy, cols_ignore=[], cols_select=[], fill_value=None):
        super().__init__(cols_select=cols_select, cols_ignore=cols_ignore)
        self.skmodel = imp.SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.strategy = strategy
        self.fill_value = fill_value


class StandardScaler(PandasTransformer):
    """ StandardScaler for pandas DFs"""

    def __init__(self, cols_ignore=[], cols_select=[]):
        super().__init__(cols_select=cols_select, cols_ignore=cols_ignore)
        self.skmodel = pre.StandardScaler()
