# Jaeger — Distributed Tracing System

Jaeger is an open-source distributed tracing system for monitoring and troubleshooting microservices. It visualizes request flows across services to identify latency bottlenecks and errors.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Span** | A single unit of work (e.g., DB query, HTTP call) with start time, duration, tags, and logs |
| **Trace** | A tree of spans forming the full path of a request across services |
| **Sampling** | Strategy for which traces to capture (probabilistic, rate-limiting, adaptive) |
| **Storage Backend** | Where trace data is persisted (Elasticsearch, Cassandra, Badger, Kafka) |
| **UI** | Web interface for searching, viewing, and comparing traces |
| **Service Dependency Graph** | Auto-generated topology showing inter-service communication and latency |

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌───────────┐
│  Service  │───▶│  Jaeger Agent │───▶│  Jaeger   │───▶│  Storage  │
│  (Client) │    │  (daemonset)  │    │  Collector │    │ (ES/Cass) │
└──────────┘    └──────────────┘    └────────────┘    └───────────┘
                                                              │
                                                              ▼
                                                       ┌───────────┐
                                                       │  Jaeger   │
                                                       │  Query UI  │
                                                       └───────────┘
```

## Deployment on Kubernetes

```bash
# Deploy Jaeger operator
kubectl create namespace observability
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/latest/download/jaeger-operator.yaml -n observability

# Create a Jaeger instance
kubectl apply -f - <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: observability
spec:
  strategy: streaming  # production: streaming via Kafka
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
        num-shards: 3
        num-replicas: 1
  ingress:
    enabled: true
EOF
```

## Integration: Request Tracing for Model Serving Pipelines

Trace an ML inference request: client → API gateway → auth → model server → feature store → response.

### Python Instrumentation with OpenTelemetry + Jaeger

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from flask import Flask, request, jsonify

import requests
import time

# Configure Jaeger exporter
resource = Resource(attributes={
    SERVICE_NAME: "model-gateway"
})
provider = TracerProvider(resource=resource)
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent.observability",
    agent_port=6831,
)
provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(provider)

# Auto-instrument Flask and requests
app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()

tracer = trace.get_tracer("model-gateway")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Create a custom span for feature preprocessing
    with tracer.start_as_current_span("feature_fetch") as span:
        span.set_attribute("feature.count", len(data.get("features", [])))
        features = fetch_features(data["features"])
        span.set_attribute("feature.store", "redis")

    # Call the model server
    with tracer.start_as_current_span("model_inference") as span:
        resp = requests.post(
            "http://torchserve.ml-serve:8080/predictions/model",
            json=features,
            timeout=5.0,
        )
        span.set_attribute("model", "bert-large")
        span.set_attribute("model.version", "v2")
        span.set_attribute("http.status_code", resp.status_code)
        span.set_attribute("inference.latency_ms", resp.elapsed.total_seconds() * 1000)
        result = resp.json()

    return jsonify(result)

def fetch_features(keys):
    # Simulated feature store call
    time.sleep(0.05)
    return {"input_ids": [101, 102, 103]}

if __name__ == "__main__":
    app.run(port=5000)
```

### Tracing Model Server Itself

```python
# instrumented_torchserve.py — trace inside the model
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def handle(data, context):
    with tracer.start_as_current_span("model_forward") as span:
        span.set_attribute("batch_size", data.shape[0])
        span.set_attribute("model", context.model_name)

        with tracer.start_as_current_span("tokenize"):
            tokens = tokenizer(data, padding=True, return_tensors="pt")

        with tracer.start_as_current_span("inference"):
            outputs = model(**tokens)

        with tracer.start_as_current_span("postprocess"):
            result = postprocess(outputs)

        span.set_attribute("output.shape", str(result.shape))
        return result
```

## Sampling Strategies

```yaml
# jaeger-sampling.yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
spec:
  strategy: allInOne
  sampling:
    options:
      default_strategy:
        type: probabilistic
        param: 0.1  # 10% of all traces
      service_strategies:
        - service: model-gateway
          type: rate_limiting
          param: 100  # max 100 traces/sec
        - service: torchserve
          type: probabilistic
          param: 0.5  # 50% of inference traces
```

## Service Dependency Graph

Query the Jaeger API to build dependency diagrams:

```bash
# Fetch dependencies
curl http://jaeger-query:16686/api/dependencies?endTs=1716336000&lookback=1h

# Returns edges between services with request count and error rate
```

## Querying Traces via API

```python
import requests
import json

jaeger_api = "http://jaeger-query:16686/api"

# Search traces by service and tags
resp = requests.get(f"{jaeger_api}/traces", params={
    "service": "model-gateway",
    "operation": "model_inference",
    "tags": json.dumps({"model": "bert-large"}),
    "limit": 20,
    "lookback": "1h",
})

# Get a single trace by ID
trace = requests.get(f"{jaeger_api}/traces/{trace_id}")

# Get services
services = requests.get(f"{jaeger_api}/services")
```

## Latency Bottleneck Analysis

```python
# latency_analysis.py — extract p50/p95/p99 from Jaeger
from datetime import datetime, timedelta
import requests

JAEGER = "http://jaeger-query:16686/api"

def analyze_latency(service: str, operation: str, lookback_minutes: int = 60):
    end = int(datetime.now().timestamp() * 1e6)  # microseconds
    start = int((datetime.now() - timedelta(minutes=lookback_minutes)).timestamp() * 1e6)

    resp = requests.get(f"{JAEGER}/traces", params={
        "service": service,
        "operation": operation,
        "start": start,
        "end": end,
        "limit": 1000,
    })
    traces = resp.json().get("data", [])

    durations = []
    for trace in traces:
        for span in trace["spans"]:
            if span["operationName"] == operation:
                durations.append(span["duration"] / 1000)  # μs → ms

    if not durations:
        return {}

    durations.sort()
    n = len(durations)
    return {
        "p50_ms": durations[int(n * 0.5)],
        "p95_ms": durations[int(n * 0.95)],
        "p99_ms": durations[int(n * 0.99)],
        "count": n,
    }

# Usage
print(analyze_latency("torchserve", "model_forward", 30))
```

## Best Practices

- Use adaptive sampling for production — sample more during failures, less during normal operation
- Add `span.set_attribute("error", True)` on exceptions to mark error spans
- Tag spans with model version, batch size, and deployment environment for filtering
- Use Jaeger streaming (Kafka) for high-throughput inference pipelines to avoid data loss
- Set appropriate storage TTL (Elasticsearch ILM: 7 days for traces, 30 days for dependencies)
- Combine with OpenTelemetry Collector as the ingestion gateway for vendor-agnostic pipelines

## Resources

- [Jaeger Docs](https://www.jaegertracing.io/docs/)
- [Jaeger Operator](https://github.com/jaegertracing/jaeger-operator)
- [OpenTelemetry + Jaeger](https://opentelemetry.io/docs/languages/python/exporters/#jaeger)
- [Sampling Guide](https://www.jaegertracing.io/docs/latest/sampling/)
