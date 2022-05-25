import os

import pandas as pd
from sklearn.model_selection import train_test_split


def submit(subm_path, ids, y_pred):
    subm_df = pd.DataFrame({
        'PassengerId': ids,
        'Survived': y_pred
    })
    if not os.path.exists(os.path.dirname(subm_path)):
        dirname = os.path.dirname(subm_path)
        print(f'Do you want to create <{dirname}> ? (Y or N)')
        s = input()
        if s == 'Y':
            os.makedirs(dirname)
        elif s == 'N':
            print('Submit canceled')
            return
    subm_df.to_csv(subm_path, index=False)


def get_lastcommit_infos(git_repo):

    def format_commit_datetime(commit_datetime):
        commit_day = commit_datetime.strftime('%d')
        commit_month = commit_datetime.strftime('%B')[:3]
        commit_year = commit_datetime.strftime('%Y')
        commit_time = commit_datetime.strftime('%H:%M')
        return f'{commit_day} {commit_month} {commit_year} at {commit_time}'

    return {
        'commit_time': format_commit_datetime(git_repo.head.commit.committed_datetime),
        'commit_message': git_repo.head.commit.message,
        'commit_sha': git_repo.head.commit.hexsha[:7],
        'Branch': git_repo.active_branch.name
    }


class CVSplitter:
    def __init__(self, n_splits, test_size):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        return [train_test_split(range(X.shape[0]), test_size=self.test_size) for _ in
                range(self.n_splits)]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
