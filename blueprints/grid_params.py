

ml_params_distributions = {
    'Clip Outliers__Float__std_band': [3.],
    'LogisticRegression__C': uniform(0.1, 3.),
}

rs = RandomizedSearchCV (
    pipe,
    param_distributions = ml_params_distributions,
    n_iter=20,
    n_jobs=-1,
    refit=True,
    cv=CVSplitter(5, 80),
    return_train_score=True
)
