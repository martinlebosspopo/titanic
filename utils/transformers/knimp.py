from sklearn.pipeline import Pipeline

from .base import ColumnsAsParameters
from .sk4pandas import OneHotEncoder, KNNImputer


class KNImp(ColumnsAsParameters):
    OH_DTYPES = ['object', 'category']

    def __init__(self, cols_ignore=[], cols_select=[], min_frequency=10, cols_onehot=[]):
        super().__init__(cols_select=cols_select, cols_ignore=cols_ignore)
        self.min_frequency = min_frequency
        self.cols_onehot = cols_onehot
        self.pipe = None

    def fit(self, X, y=None):
        super().fit(X)

        X_cols_final = X[self.cols_final]
        if not self.cols_onehot:
            self.cols_onehot = [col for col, dtype in zip(X_cols_final.columns, X_cols_final.dtypes)
                                if dtype in self.OH_DTYPES]
        print('<KNImputer> Cols that will be OneHot encoded :')
        for col in self.cols_onehot:
            print('\t-', col, set(X[col]))

        self.pipe = Pipeline([
            ('One HOT', OneHotEncoder(
                cols_select=self.cols_onehot,
                keep_nans=True,
                min_frequency=self.min_frequency)),
            ('KNN imp', KNNImputer(cols_ignore=self.cols_ignore))
        ])

        self.pipe.fit(X)
        self.pipe[1].group_features = self.pipe[0].get_grouped_features()
        return self

    def transform(self, X, y=None):
        X_transformed = self.pipe.transform(X)
        return self.pipe[0].inverse_transform(X_transformed)
