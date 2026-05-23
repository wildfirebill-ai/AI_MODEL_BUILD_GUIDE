# XGBoost — Gradient Boosting Framework

Optimized gradient boosting for tabular/structured data. Scikit-learn and native APIs.

## Key Components

| Component | Description |
|-----------|-------------|
| `xgboost.XGBClassifier` / `XGBRegressor` | Scikit-learn API |
| `xgboost.train()` | Native API with custom objectives |
| `xgboost.DMatrix` | Internal data structure |
| Tree methods `hist`, `approx`, `exact` | Split-finding algorithms |
| `tree_method='gpu_hist'` | GPU acceleration |
| `early_stopping_rounds` | Early stopping in `fit()` / `cv()` |

## Installation

```bash
pip install xgboost
# GPU (CUDA required)
pip install xgboost[cuda]
```

## Scikit-learn API

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    eval_metric='logloss', early_stopping_rounds=10, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

## Native API

```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'binary:logistic', 'max_depth': 6, 'eta': 0.1, 'tree_method': 'hist'}
model = xgb.train(params, dtrain, num_boost_round=100,
                  evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=10)
```

## Feature Importance & GPU

```python
xgb.plot_importance(model, importance_type='gain')

# GPU acceleration
model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
```

## Cross-Validation & Tuning

```python
cv_results = xgb.cv(params, dtrain, nfold=5, num_boost_round=200, early_stopping_rounds=10)
print(f"Best CV: {cv_results['test-logloss-mean'].min():.4f}")

import optuna
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    model = xgb.XGBClassifier(**param, n_estimators=100, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model.score(X_test, y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f"Best params: {study.best_params}")
```

## Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pipe = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), ['age', 'fare']),
        ('cat', OneHotEncoder(), ['sex', 'embarked']),
    ])),
    ('clf', xgb.XGBClassifier(n_estimators=200)),
])
pipe.fit(X_train, y_train)
```

## Resources

- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [GitHub](https://github.com/dmlc/xgboost)
- [Parameter Reference](https://xgboost.readthedocs.io/en/latest/parameter.html)
