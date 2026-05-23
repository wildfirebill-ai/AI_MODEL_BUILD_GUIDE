# SHAP — SHapley Additive exPlanations

Game-theoretic feature importance for model interpretability. Supports trees, deep learning, and any black-box model.

## Key Components

| Component | Description |
|-----------|-------------|
| `shap.TreeExplainer` | Fast exact SHAP for XGBoost, LightGBM, RF |
| `shap.DeepExplainer` | Deep learning (PyTorch, TensorFlow) |
| `shap.KernelExplainer` | Model-agnostic black-box explainer |
| `shap.summary_plot` | Global importance + effect direction |
| `shap.force_plot` | Per-prediction explanation |
| `shap.dependence_plot` | Feature interaction visualization |

## Installation

```bash
pip install shap
```

## Tree Explainer

```python
import shap
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
names = load_breast_cancer().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

## Summary Plot

```python
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, X_test, feature_names=names, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)
```

## Force & Waterfall Plots

```python
# Force plot for a single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0],
                feature_names=names, matplotlib=True, show=False)
plt.savefig('shap_force.png', dpi=150)

# Waterfall decomposition
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0], base_values=explainer.expected_value,
    data=X_test[0], feature_names=names))
```

## Dependence Plot

```python
shap.dependence_plot(0, shap_values, X_test, feature_names=names,
                     interaction_index=7, show=False)
plt.savefig('shap_dependence.png', dpi=150)
```

## Deep Explainer — Neural Networks

```python
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 32),
                    nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
net.load_state_dict(torch.load('model.pt'))

explainer = shap.DeepExplainer(net, torch.tensor(X_train[:100], dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(X_test[:10], dtype=torch.float32))
```

## Kernel Explainer — Model-Agnostic

```python
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[:10], nsamples=100)
```

## Bar Plot & Pipeline Integration

```python
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=names, show=False)

def explain_model(model, X, model_type='tree'):
    explainer = {
        'tree': shap.TreeExplainer,
        'deep': lambda m: shap.DeepExplainer(m, X[:100]),
    }.get(model_type, shap.KernelExplainer)(model)
    return explainer.shap_values(X)
```

## Resources

- [SHAP Docs](https://shap.readthedocs.io/)
- [GitHub](https://github.com/shap/shap)
- [NIPS Paper](https://arxiv.org/abs/1705.07874)
