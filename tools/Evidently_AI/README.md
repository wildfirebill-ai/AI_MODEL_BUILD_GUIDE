# Evidently AI — ML Monitoring & Observability

Detects data drift, model performance degradation, and data quality issues in production.

## Key Components

| Component | Description |
|-----------|-------------|
| `evidently.Report` | Interactive monitoring reports |
| `evidently.test_suite` | Automated tests against data/model |
| `evidently.metric_preset` | Pre-configured metric groups |
| Data Drift | Statistical tests (KS, chi-squared, JS divergence) |
| Column Mapping | Feature type mappings |

## Installation

```bash
pip install evidently
```

## Data Drift Detection

```python
import pandas as pd
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

data = load_iris(as_frame=True).frame
data.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in data.columns]
reference, current = data.iloc[:100], data.iloc[100:]

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html("data_drift.html")
```

## Classification Performance

```python
from evidently.metric_preset import ClassificationPreset

data["prediction"] = 1
data["target"] = data["target"].astype(int)

report = Report(metrics=[ClassificationPreset()])
report.run(reference_data=data.iloc[:100], current_data=data.iloc[100:])
report.save_html("classification_perf.html")
```

## Regression Performance

```python
from evidently.metric_preset import RegressionPreset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame({"x1": range(200), "x2": [x * 0.5 + 2 for x in range(200)]})
df["target"] = df["x1"] * 0.8 + df["x2"] * 0.2 + 5
X_train, X_test, y_train, y_test = train_test_split(df[["x1", "x2"]], df["target"], test_size=0.3)

model = LinearRegression().fit(X_train, y_train)
ref = pd.DataFrame({"x1": X_train["x1"], "x2": X_train["x2"],
                     "target": y_train, "prediction": model.predict(X_train)})
cur = pd.DataFrame({"x1": X_test["x1"], "x2": X_test["x2"],
                     "target": y_test, "prediction": model.predict(X_test)})

report = Report(metrics=[RegressionPreset()])
report.run(reference_data=ref, current_data=cur)
```

## Test Suite

```python
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestNumberOfMissingValues, TestValueRange

suite = TestSuite(tests=[
    TestColumnDrift("sepal_length"), TestNumberOfMissingValues(),
    TestValueRange("sepal_length", left=4, right=8),
])
suite.run(reference_data=reference, current_data=current)
suite.save_html("test_suite.html")
```

## Column Mapping

```python
from evidently.pipeline.column_mapping import ColumnMapping

mapping = ColumnMapping(
    target="target", prediction="prediction",
    numerical_features=["sepal_length", "sepal_width"],
)
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current, column_mapping=mapping)
```

## Data Quality & Combined Dashboards

```python
from evidently.metric_preset import DataQualityPreset, TargetDriftPreset

Report(metrics=[DataQualityPreset()]).run(reference_data=reference, current_data=current)
Report(metrics=[TargetDriftPreset()]).run(reference_data=ref, current_data=cur)

# Combined dashboard
Report(metrics=[DataDriftPreset(), DataQualityPreset(), ClassificationPreset()])
```

## Resources

- [Evidently Docs](https://docs.evidentlyai.com/)
- [GitHub](https://github.com/evidentlyai/evidently)
- [Presets Reference](https://docs.evidentlyai.com/reference/api-reference/evidently/metrics/preset)
