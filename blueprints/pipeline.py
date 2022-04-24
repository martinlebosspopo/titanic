

clip_outliers = [
    (
        'Float',
        tr.ClipOutliers(std_band=3),
        make_column_selector(dtype_include=['float64'])
    )
]

fillna = [
    (
        'Mean',
        s4p.SimpleImputer(strategy='mean'),
        make_column_selector(dtype_include=['float64'])
    ),
    (
        'Most Frequent',
        s4p.SimpleImputer(strategy='most_frequent'),
        make_column_selector(dtype_include=['int64', 'object'])
    )
]

standardize = [
    (
        'Floats Ints',
        s4p.StandardScaler(),
        ['Age', 'Fare']
    )
]


steps_main = [
    ('Format Cabins', tr.Cabin() ),
    ('Set working columns', tr.SetupFeatures(cols_ignore=['PassengerId', 'Name', 'Ticket', 'Parch']) ),
    ('Clip Outliers', s4p.ColumnTransformer(clip_outliers, remainder='passthrough') ),
    ('Standardize', s4p.ColumnTransformer(standardize, remainder='passthrough')),
    ('Fill NaN', s4p.ColumnTransformer(fillna, remainder='passthrough') ),
    ('Convert Types', tr.AsTypes(coltypes_overwrite={'Age': 'float64'}) ),
    ('One Hot', s4p.OneHotEncoder(cols_select=['Pclass', 'Sex', 'Cabin', 'SibSp', 'Embarked']) ),
    ('Decision Tree', DecisionTreeClassifier())
]

pipe = Pipeline(steps_main)
