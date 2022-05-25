from .base import BaseTransformer, ColumnsAsParameters

from sklearn.preprocessing import FunctionTransformer


class Cabin(BaseTransformer):
    def __init__(self):
        self.cabins_known = None

    def get_cabins(self, X):
        return X['Cabin'].str.extract('(.).*')[0]

    def fit(self, X, y=None):
        self.cabins_known = set(self.get_cabins(X))
        return self

    def transform(self, X, y=None):
        cabins = self.get_cabins(X)
        unknown_cabins = set(cabins) - self.cabins_known
        X['Cabin'] = cabins.replace(unknown_cabins, float('NaN'))
        return X


class SetupFeatures(ColumnsAsParameters):
    def transform(self, X, y=None):
        missing_cols = [col for col in self.cols_final if col not in X.columns]
        supp_cols = [col for col in X.columns if col not in self.cols_final]
        X = X.set_index(supp_cols)
        X[missing_cols] = float('NaN')
        return X[self.cols_final]


class ClipOutliers(ColumnsAsParameters):
    def __init__(self, std_band=2.5, cols_ignore=[], cols_select=[]):
        super().__init__(cols_ignore=cols_ignore, cols_select=cols_select)
        self.means = None
        self.stds = None
        self.std_band = std_band

    def fit(self, X, y=None):
        super().fit(X, y=y)
        self.means = X[self.cols_final].mean(numeric_only=True)
        self.stds = X[self.cols_final].std(numeric_only=True)
        return self

    def transform(self, X, y=None):
        X[self.cols_final] = X[self.cols_final].clip(
            lower=self.means - self.std_band * self.stds,
            upper=self.means + self.std_band * self.stds,
            axis=1,
        )
        return X


class AsTypes(FunctionTransformer):
    COL_TYPES = {
        'Pclass': 'category',
        'Sex': 'category',
        'Age': 'float',
        'SibSp': 'category',
        'Parch': 'category',
        'Fare': 'float64',
        'Cabin': 'category',
        'Embarked': 'category'
    }

    def __init__(self, coltypes_overwrite={}):
        super().__init__()
        self.coltypes_overwrite = coltypes_overwrite
        self.func = self.cust_func(coltypes_overwrite)

    def cust_func(self, coltypes_overwrite):
        col_types = AsTypes.COL_TYPES.copy()
        col_types.update(coltypes_overwrite)

        def func(X):
            for col, dtype in col_types.items():
                try:
                    X = X.astype({col: dtype})
                except KeyError as err:
                    pass
            return X

        return func