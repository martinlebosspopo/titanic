

ml_params_distributions = {
    'Random Forest__n_estimators': [10, 20, 30, 40, 50, 70, 90, 120, 150, 180, 190, 200],
    'Random Forest__max_depth': ['None', 5, 10],
    'Random Forest__min_samples_split': [2, 10, 20, 30],
    'Random Forest__max_features': ['sqrt', 'sqrt', 'log2', 0.2, 0.3],
    'Random Forest__bootstrap': [True, False],
    'Random Forest__max_samples': [0.2, 0.4, 0.6, 0.8]
}

rs = RandomizedSearchCV (
    pipe,
    param_distributions = ml_params_distributions,
    n_iter=1,
    n_jobs=-1,
    refit=True,
    cv=CVSplitter(5, 80),
    return_train_score=True
)
