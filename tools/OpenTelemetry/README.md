# OpenTelemetry — Cloud-Native Observability Framework

OpenTelemetry (OTel) is a vendor-neutral observability standard for collecting **traces**, **metrics**, and **logs** from cloud-native applications. It provides APIs, SDKs, and a Collector for telemetry ingestion, processing, and export.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Traces** | End-to-end request path across distributed services (spans with parent-child relationships) |
| **Metrics** | Aggregated numerical measurements (counters, gauges, histograms) |
| **Logs** | Structured or unstructured event records with severity levels |
| **OpenTelemetry Collector** | Vendor-agnostic agent/gateway for receiving, processing, and exporting telemetry |
| **Exporter** | Sends telemetry to backends (Jaeger, Prometheus, Datadog, Grafana) |
| **Auto-Instrumentation** | Zero-code injection of telemetry via agent or eBPF |

## Architecture

```
┌────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Application│───▶│ OTel Collector    │───▶│ Backend (Jaeger) │
│ (SDK/Agent) │    │ (pipeline:        │    │ Prometheus        │
└────────────┘    │  receive→process   │    │ Grafana           │
                  │  →export)         │    └─────────────────┘
                  └──────────────────┘
```

## Setup: Collector Configuration

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  memory_limiter:
    check_interval: 1s
    limit_mib: 512

exporters:
  jaeger:
    endpoint: "jaeger:14250"
    tls:
      insecure: true
  prometheus:
    endpoint: "0.0.0.0:8889"
  debug:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger, debug]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus, debug]
```

## Integration: Distributed Tracing for Model Serving

Trace a model inference request end-to-end: client → API gateway → model server → feature store.

### Python Auto-Instrumentation

```bash
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap --action=install
```

```python
# app.py — instrumented FastAPI model server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

import fastapi
import requests

# Configure tracer
resource = Resource(attributes={
    SERVICE_NAME: "torchserve-model-server"
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://otel-collector:4317")
))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

app = fastapi.FastAPI()
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

@app.post("/predict")
async def predict(features: dict):
    with tracer.start_as_current_span("preprocess") as span:
        span.set_attribute("input.size", len(features))
        processed = preprocess(features)

    with tracer.start_as_current_span("model_inference") as span:
        # Call model server
        resp = requests.post("http://triton:8001/infer", json=processed)
        span.set_attribute("model", "bert-large")
        span.set_attribute("latency_ms", resp.elapsed.total_seconds() * 1000)
        result = resp.json()

    with tracer.start_as_current_span("postprocess") as span:
        output = postprocess(result)
        span.set_attribute("output.class", output.get("label"))

    return output
```

### Kubernetes Sidecar Deployment

```yaml
# deployment.yaml — inject OTel sidecar
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
  labels:
    app: torchserve
spec:
  replicas: 2
  selector:
    matchLabels:
      app: torchserve
  template:
    metadata:
      labels:
        app: torchserve
    spec:
      containers:
        - name: model-server
          image: pytorch/torchserve:0.9.0
          ports:
            - containerPort: 8080
          env:
            - name: OTEL_SERVICE_NAME
              value: "torchserve"
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://localhost:4317"
        - name: otel-sidecar
          image: otel/opentelemetry-collector-contrib:latest
          ports:
            - containerPort: 4317
            - containerPort: 13133  # health check
          volumeMounts:
            - name: otel-config
              mountPath: /etc/otelcol-contrib/config.yaml
              subPath: config.yaml
      volumes:
        - name: otel-config
          configMap:
            name: otel-collector-config
```

## Custom Metrics for Model Performance

```python
from opentelemetry.metrics import get_meter

meter = get_meter("ml-monitoring", "1.0.0")
inference_latency = meter.create_histogram(
    name="model.inference.latency",
    description="Model inference latency in milliseconds",
    unit="ms",
)
prediction_counter = meter.create_counter(
    name="model.predictions.total",
    description="Total number of predictions",
)
model_version = meter.create_up_down_counter(
    name="model.version.active",
    description="Currently active model version",
)

def predict(features):
    start = time.time()
    result = model.infer(features)
    latency = (time.time() - start) * 1000

    inference_latency.record(latency, {"model": "bert", "version": "v2"})
    prediction_counter.add(1, {"model": "bert", "outcome": "success"})
    return result
```

## Samplers for High-Volume Inference

```yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 10  # Sample 10% of inference traces
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [probabilistic_sampler, batch]
      exporters: [jaeger]
```

## Best Practices

- Use the OTel Collector as a gateway — never export directly from apps in prod
- Set `OTEL_RESOURCE_ATTRIBUTES` with `service.name`, `service.version`, `deployment.environment`
- Enable tail-based sampling for high-volume inference workloads
- Use `memory_limiter` processor to prevent OOM in the collector
- Batch spans and metrics before exporting to reduce network overhead
- Pair with Jaeger for traces and Prometheus/Grafana for metrics dashboards

## Resources

- [OpenTelemetry Docs](https://opentelemetry.io/docs/)
- [Collector Configuration](https://opentelemetry.io/docs/collector/configuration/)
- [Python SDK](https://opentelemetry.io/docs/languages/python/)
- [Auto-Instrumentation](https://opentelemetry.io/docs/languages/python/automatic/)
