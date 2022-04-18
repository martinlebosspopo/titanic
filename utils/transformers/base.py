from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, *args):
        print('GET FEATURES NAMES', args)
        return args[0]


class ColumnsAsParameters(BaseTransformer):
    def __init__(self, cols_ignore=[], cols_select=[]):
        self.cols_ignore = cols_ignore
        self.cols_select = cols_select
        self.cols_final = None

    def fit(self, X, y=None):
        self.cols_final = self.cols_select or [col for col in X.columns if
                                               col not in self.cols_ignore]
        return self
