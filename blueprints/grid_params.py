
n_iter = 50

ml_params_distributions = {
    'Decision Tree__min_samples_split': loguniform(40, 100).rvs(n_iter).astype('int'),
}

rs = RandomizedSearchCV (
    pipe,
    param_distributions = ml_params_distributions,
    n_iter=n_iter,
    n_jobs=-1,
    refit=True,
    cv=CVSplitter(5, 80),
    return_train_score=True
)
