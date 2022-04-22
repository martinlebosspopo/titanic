# Comment

ml_params_distributions = {
    'Logistic Regression__C': uniform(1, 10),
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
