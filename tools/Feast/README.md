# Feast — Feature Store for ML

Feast is an open-source feature store that enables consistent feature definition, computation, and serving across training and inference pipelines.

## Key Concepts

- **Feature View** — a group of features from a single data source
- **Feature Service** — a logical grouping of feature views for serving
- **Entity** — a domain object with a key (e.g. `driver_id`)
- **Online Store** — low-latency store (Redis, DynamoDB) for serving
- **Offline Store** — batch store (BigQuery, Snowflake) for training data

## Defining Features

```python
from datetime import timedelta
from feast import Entity, FeatureView, FeatureService, Field
from feast.types import Float32, Int64

driver = Entity(name="driver_id", description="Driver identifier", value_type=Int64)

driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="avg_daily_trips", dtype=Float32),
        Field(name="driver_rating", dtype=Float32),
    ],
    source=BigQuerySource(query="SELECT * FROM feast.driver_stats"),
)

service = FeatureService(name="driver_activity", features=[driver_stats_fv])
```

## Apply and Deploy

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
store.apply()
```

## Retrieving Training Data (Offline)

```python
import pandas as pd

entity_df = pd.DataFrame({
    "driver_id": [1001, 1002, 1003],
    "event_timestamp": [pd.Timestamp.now()] * 3,
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_hourly_stats:avg_daily_trips",
              "driver_hourly_stats:driver_rating"],
).to_df()
```

## Online Serving (Inference)

```python
store.materialize_incremental(end_date=pd.Timestamp.now())

features = store.get_online_features(
    features=["driver_hourly_stats:avg_daily_trips",
              "driver_hourly_stats:driver_rating"],
    entity_rows=[{"driver_id": 1001}],
).to_dict()
```

## Point-in-Time Joins

```python
entity_df = pd.DataFrame({
    "driver_id": [1001, 1001, 1002],
    "event_timestamp": [
        pd.Timestamp("2024-06-01 08:00:00"),
        pd.Timestamp("2024-06-01 09:00:00"),
        pd.Timestamp("2024-06-01 08:00:00"),
    ],
})
features = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_hourly_stats:avg_daily_trips"],
).to_df()
```

## Integration: Training Pipeline

```python
def generate_training_set(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    features = store.get_historical_features(
        entity_df=entity_df, features=["driver_activity"],
    ).to_df()
    features.drop(columns=["event_timestamp"], inplace=True)
    return features

store = FeatureStore(repo_path="feature_repo")
store.apply()
X = generate_training_set(store, pd.read_parquet("labels.parquet"))
y = pd.read_parquet("labels.parquet")["target"]
```

## Best Practices

- Keep feature views small (≤50 features) and logically grouped.
- Materialize features to online store before production deployment.
- Use point-in-time joins to avoid data leakage in training data.
- Use feature services to simplify client code for inference.
