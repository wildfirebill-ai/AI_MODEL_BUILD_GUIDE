# Kibana — Data Visualization for Elasticsearch

Kibana is the visualization and exploration layer for Elasticsearch. It provides dashboards, charts, alerting, and built-in ML for log/metric analysis across any Elasticsearch-indexed data.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Dashboards** | Collections of visualizations (charts, tables, maps) for operational views |
| **Discover** | Ad-hoc log/metric exploration with Lucene/KQL search and filtering |
| **Lens** | Drag-and-drop chart builder with automatic visualization type selection |
| **Canvas** | Pixel-perfect infographic-style reports for executive presentations |
| **Alerts** | Rule-based alerting (threshold, anomaly, frequency) with Slack/PagerDuty/webhook actions |
| **Elastic ML** | Built-in machine learning (single/multi-metric, population, rare, forecast) |
| **Logs & Metrics** | Pre-built UIs for exploring structured logs (Elastic Common Schema) and infrastructure metrics |

## Architecture

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐
│  Data     │───▶│ Elasticsearch│───▶│   Kibana    │
│  Sources  │    │  (storage +   │    │  (visualize, │
│  (Beats,  │    │   search)    │    │   manage)   │
│   OTel)   │    └─────────────┘    └──────┬──────┘
└──────────┘                               │
                                           ▼
                                   ┌───────────────┐
                                   │  Users (Browser)│
                                   └───────────────┘
```

## Setup

```bash
# Deploy Elasticsearch + Kibana via Helm
helm repo add elastic https://helm.elastic.co
helm install es elastic/elasticsearch --namespace observability --set replicas=3
helm install kb elastic/kibana --namespace observability \
  --set elasticsearchHosts=http://elasticsearch-master:9200

# Port-forward for local access
kubectl port-forward svc/kb-kibana 5601 -n observability

# Or via Docker
docker run -d --name kibana -p 5601:5601 \
  -e ELASTICSEARCH_HOSTS=http://host.docker.internal:9200 \
  docker.elastic.co/kibana/kibana:8.12.0
```

## Integration: Log Analytics for Model Serving

Send model serving logs to Elasticsearch via Filebeat or OTel Collector, then visualize in Kibana.

### Ingest Model Logs via Filebeat

```yaml
# filebeat-config.yaml
filebeat.inputs:
  - type: container
    paths:
      - "/var/log/containers/*.log"
    processors:
      - add_kubernetes_metadata:
          host: ${HOSTNAME}
      - dissect:
          tokenizer: "%{timestamp} [%{level}] %{model_name}: %{message}"
          target_prefix: "model"

output.elasticsearch:
  hosts: ["${ELASTICSEARCH_HOST:elasticsearch:9200}"]
  index: "model-logs-%{+yyyy.MM.dd}"

setup.template.name: "model-logs"
setup.template.pattern: "model-logs-*"
```

### Ingest via OpenTelemetry Collector

```yaml
# otel-collector.yaml — send logs to Elasticsearch
exporters:
  elasticsearch:
    endpoints: ["http://elasticsearch:9200"]
    logs_index: "otel-model-logs"
    mapping:
      mode: ecs

service:
  pipelines:
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [elasticsearch]
```

### Python Structured Logging to Elasticsearch

```python
import logging
import json
from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch("http://elasticsearch:9200")

class ModelLogger:
    def __init__(self, es_client):
        self.es = es_client

    def log_prediction(self, model_name, version, latency_ms, status,
                       input_shape, prediction):
        doc = {
            "@timestamp": datetime.utcnow().isoformat(),
            "model": {
                "name": model_name,
                "version": version,
            },
            "inference": {
                "latency_ms": latency_ms,
                "status": status,
                "input_shape": input_shape,
            },
            "prediction": prediction,
            "environment": os.getenv("DEPLOY_ENV", "dev"),
        }
        self.es.index(index="model-predictions", document=doc)

logger = ModelLogger(es)
logger.log_prediction(
    model_name="bert-large",
    version="v2.1.0",
    latency_ms=247.3,
    status="success",
    input_shape=[1, 512],
    prediction={"label": "POSITIVE", "score": 0.997},
)
```

## Kibana Dashboards for ML

### Dashboard Panels

| Panel | Query | Visualization |
|-------|-------|---------------|
| Requests per minute | `model.name: "bert-large"` | Line chart (Lens) |
| P50/P95/P99 latency | `model.name: "bert-large"` | Percentile rank aggregation |
| Error rate | `inference.status: "error"` | Pie chart or metric tile |
| Model version distribution | `model.version: *` | Data table or treemap |
| Top active models | `*` | Tag cloud (count by `model.name`) |
| Latency heatmap | `model.name: *` | Heatmap (x: time, y: model) |

### Create an Alert for High Latency

```json
// .kibana/alert — PUT to create rule via API
POST api/alerting/rule
{
  "name": "Model Latency Spike",
  "rule_type_id": "threshold",
  "consumer": "alerts",
  "params": {
    "index": ["model-predictions"],
    "timeField": "@timestamp",
    "aggType": "avg",
    "aggField": "inference.latency_ms",
    "groupBy": "top",
    "termSize": 10,
    "termField": "model.name.keyword",
    "thresholdComparator": ">",
    "threshold": [500],
    "timeWindowSize": 5,
    "timeWindowUnit": "m"
  },
  "actions": [
    {
      "group": "threshold met",
      "id": "<slack-connector-id>",
      "params": {
        "message": "Latency spike detected for {{context.term}} — {{context.value}}ms avg"
      }
    }
  ]
}
```

## Elastic ML for Anomaly Detection

```yaml
# Create ML job via API
POST _ml/anomaly_detectors/model-latency
{
  "description": "Model inference latency anomaly detection",
  "analysis_config": {
    "bucket_span": "5m",
    "detectors": [
      {
        "function": "high_mean",
        "field_name": "inference.latency_ms",
        "partition_field_name": "model.name",
        "detector_description": "High latency by model"
      }
    ]
  },
  "data_description": {
    "time_field": "@timestamp",
    "time_format": "epoch_ms"
  }
}

POST _ml/datafeeds/datafeed-model-latency/_start
{
  "start": "now-7d"
}
```

## Saved Objects API for Dashboard as Code

```python
# export_dashboard.py — export Kibana dashboards programmatically
import requests

KIBANA = "http://localhost:5601"
headers = {"kbn-xsrf": "true"}

# Export all saved objects
resp = requests.post(
    f"{KIBANA}/api/saved_objects/_export",
    headers=headers,
    json={
        "type": ["dashboard", "visualization", "lens", "search"],
        "excludeExportDetails": False,
    },
)

with open("model-dashboards.ndjson", "wb") as f:
    f.write(resp.content)

# Import on another cluster
with open("model-dashboards.ndjson", "rb") as f:
    requests.post(
        f"{KIBANA}/api/saved_objects/_import?overwrite=true",
        headers=headers,
        files={"file": f},
    )
```

## Best Practices

- Use Elastic Common Schema (ECS) for log formatting — Kibana's built-in Logs UI relies on it
- Create Index Lifecycle Policies (ILM) to manage log retention (hot→warm→cold→delete)
- Set `kibana.indexPattern` in Fleet for auto-discovery of new ML indices
- Use Lens for most visualizations — it auto-selects the best chart type and is query-performant
- Export dashboards as NDJSON and store in Git (dashboard-as-code)
- Enable spaces to separate ML team dashboards from platform/infra views
- Use the Elasticsearch Query DSL (bool/term/range) for precise aggregation filters

## Resources

- [Kibana Docs](https://www.elastic.co/guide/en/kibana/current/index.html)
- [Alerting Guide](https://www.elastic.co/guide/en/kibana/current/alerting-getting-started.html)
- [Elastic ML](https://www.elastic.co/guide/en/machine-learning/current/ml-getting-started.html)
- [Dashboard as Code](https://www.elastic.co/guide/en/kibana/current/managing-saved-objects.html)
