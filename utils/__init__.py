import pandas as pd
from sklearn.model_selection import train_test_split


def submit(subm_path, ids, y_pred):
    subm_df = pd.DataFrame({
        'PassengerId': ids,
        'Survived': y_pred
    })

    subm_df.to_csv(subm_path, index=False)


class CVSplitter:
    def __init__(self, n_splits, test_size):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        return [train_test_split(range(X.shape[0]), test_size=self.test_size) for _ in
                range(self.n_splits)]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
