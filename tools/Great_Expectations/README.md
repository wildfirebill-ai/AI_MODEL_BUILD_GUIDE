# Great Expectations — Data Validation & Quality

Declarative data validation framework with auto-generated quality reports.

## Key Components

| Component | Description |
|-----------|-------------|
| Expectations | Assertions about columns (nulls, types, ranges, distributions) |
| Suites | Collections of expectations per data asset |
| Checkpoints | Configurable validations with actions |
| Data Docs | Auto-generated HTML data quality reports |
| `ge.dataset.PandasDataset` | Pandas-backed dataset wrapper |

## Installation

```bash
pip install great_expectations
pip install great_expectations[spark]  # Spark support
pip install great_expectations[sqlalchemy]  # SQL support
```

## Quick Start

```python
import great_expectations as ge
import pandas as pd

df = pd.DataFrame({
    "passenger_id": range(1, 11),
    "name": ["Alice", "Bob", None, "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
    "age": [25, 30, 35, None, 28, 40, 22, 55, 33, 27],
    "fare": [100.0, 150.0, 200.0, 120.0, 180.0, 90.0, 110.0, 250.0, 130.0, 175.0],
    "survived": [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
})
ge_df = ge.dataset.PandasDataset(df)
```

## Defining Expectations

```python
ge_df.expect_column_to_exist("name")
ge_df.expect_column_values_to_not_be_null("passenger_id")
ge_df.expect_column_values_to_not_be_null("name")
ge_df.expect_column_values_to_be_between("age", 0, 120)
ge_df.expect_column_values_to_be_between("fare", 0, 1000)
ge_df.expect_column_values_to_be_in_set("survived", [0, 1])
ge_df.expect_column_values_to_be_unique("passenger_id")
ge_df.expect_column_values_to_be_of_type("age", "int64")
ge_df.expect_column_proportion_to_be_between("survived", 0.4, 0.6)
ge_df.expect_column_mean_to_be_between("age", 20, 50)
ge_df.expect_column_values_to_match_regex("name", r"^[A-Z][a-z]+$")
```

## Validation Results

```python
results = ge_df.validate()
print(f"Success: {results['success']}")
print(f"Statistics: {results['statistics']}")
for exp in results["results"]:
    status = "✓" if exp["success"] else "✗"
    print(f"{status} {exp['expectation_config']['expectation_type']}")
```

## Expectation Suites

```python
import json

suite = ge_df.get_expectation_suite(discard_failed_expectations=False)
suite["expectation_suite_name"] = "titanic.basic_quality"
with open("suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# Apply suite to new data
with open("suite.json") as f:
    loaded = json.load(f)
new_df = ge.dataset.PandasDataset(pd.DataFrame({
    "passenger_id": [11, 12], "name": ["Kate", "Leo"],
    "age": [29, 31], "fare": [140.0, 160.0], "survived": [1, 0],
}))
new_df.set_expectation_suite(loaded)
print(f"Valid: {new_df.validate()['success']}")
```

## Checkpoint & CI/CD Integration

```python
# Checkpoint config (YAML)
checkpoint_config = """
name: titanic_checkpoint
config_version: 1.0
class_name: SimpleCheckpoint
validations:
  - batch_request:
      datasource_name: my_datasource
      data_asset_name: titanic
    expectation_suite_name: titanic.basic_quality
"""

# CI: fail if quality below threshold
validation = ge_df.validate()
ratio = validation["statistics"]["successful_expectations"] / max(
    validation["statistics"]["evaluated_expectations"], 1)
if ratio < 0.95:
    raise ValueError(f"Quality {ratio:.1%} below threshold")
print(f"Data quality: {ratio:.1%}")
```

## Resources

- [GE Docs](https://docs.greatexpectations.io/)
- [Expectation Gallery](https://greatexpectations.io/expectations/)
- [GitHub](https://github.com/great-expectations/great_expectations)
