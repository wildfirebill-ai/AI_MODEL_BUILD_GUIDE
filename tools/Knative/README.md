# Knative — Serverless Container Platform on Kubernetes

Knative extends Kubernetes to run stateless, request-driven workloads with automatic scaling — including scale-to-zero. It is split into **Serving** (serverless workloads) and **Eventing** (event-driven architecture).

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Service** | Top-level resource that manages the full lifecycle (Configuration + Route + Revision) |
| **Revision** | An immutable snapshot of code + configuration — each deployment creates a new revision |
| **Route** | Traffic routing: split traffic across revisions (canary, blue/green) |
| **Configuration** | Desired state — automatically creates new Revisions on changes |
| **Autoscaling** | KPA (Knative Pod Autoscaler) scales based on concurrency or RPS; supports scale-to-zero |
| **Traffic Splitting** | Route percentage of requests to different revisions for A/B testing or canary deploys |

## Architecture

```
┌───────────────────────────────────────────┐
│               Knative Service              │
│  ┌─────────────┐    ┌──────────────────┐  │
│  │ Configuration│───▶│     Route        │  │
│  │              │    │ (traffic rules)  │  │
│  └──────┬───────┘    └────────┬─────────┘  │
│         │                     │            │
│         ▼                     ▼            │
│  ┌──────────────────────────────────────┐  │
│  │  Revision v1  │  Revision v2         │  │
│  │  (active)     │  (canary, 10% traffic)│  │
│  └───────────────┴──────────────────────┘  │
└───────────────────────────────────────────┘
```

## Core Service Definition

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: bert-model
  namespace: ml-serve
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"           # scale-to-zero
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/target: "5"             # concurrency target
        autoscaling.knative.dev/metric: "concurrency"   # or "rps"
    spec:
      containers:
        - image: ghcr.io/org/bert-serving:v1.2.3
          ports:
            - containerPort: 8080
          env:
            - name: MODEL_NAME
              value: "bert-base-uncased"
            - name: MAX_BATCH_SIZE
              value: "32"
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
            initialDelaySeconds: 10
            periodSeconds: 5
```

## Integration: Serverless Model Serving with Scale-to-Zero

Knative is ideal for ML inference workloads with variable traffic — scale to zero when idle, scale up instantly on request.

### Deploy and Manage

```bash
# Install Knative Serving
kubectl apply -f https://github.com/knative/serving/releases/latest/download/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/latest/download/serving-core.yaml
kubectl apply -f https://github.com/knative/net-kourier/releases/latest/download/kourier.yaml

# Deploy a model service
kubectl apply -f bert-model.yaml

# Get the service URL
kn service list

# Send a prediction
curl -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  http://bert-model.ml-serve.example.com/predict

# Watch pods scale from 0 to N on request
kubectl get pods -n ml-serve -w
```

### Traffic Splitting (Canary Deploy)

```bash
# Deploy a new revision with model v2
kn service update bert-model \
  --image ghcr.io/org/bert-serving:v2.0.0 \
  --env MODEL_NAME=bert-large-uncased

# Split traffic: 90% v1, 10% v2
kn service update bert-model \
  --traffic @latest=10 \
  --traffic bert-model-00001=90

# Promote v2 to 100% after validation
kn service update bert-model \
  --traffic @latest=100
```

### YAML for Traffic Splitting

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: bert-model
spec:
  template:
    metadata:
      name: bert-model-v2
    spec:
      containers:
        - image: ghcr.io/org/bert-serving:v2.0.0
  traffic:
    - revisionName: bert-model-v2
      percent: 10
    - revisionName: bert-model-v1
      percent: 90
    - latestRevision: false
```

### Autoscaling Configuration

```yaml
# Aggressive scale-up for burst inference traffic
metadata:
  annotations:
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "50"
    autoscaling.knative.dev/target: "10"           # 10 concurrent requests per pod
    autoscaling.knative.dev/metric: "concurrency"
    autoscaling.knative.dev/scale-down-delay: "30s" # hold pods 30s before scaling down
    autoscaling.knative.dev/window: "60s"           # metrics window for scaling decisions
```

### GPU Support for Model Serving

```yaml
spec:
  template:
    spec:
      containers:
        - image: ghcr.io/org/bert-serving:gpu
          resources:
            limits:
              nvidia.com/gpu: "1"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

## Eventing: Trigger Inference from Events

```yaml
apiVersion: eventing.knative.dev/v1
kind: Trigger
metadata:
  name: image-classify-trigger
  namespace: ml-serve
spec:
  broker: default
  filter:
    attributes:
      type: image.uploaded
  subscriber:
    ref:
      apiVersion: serving.knative.dev/v1
      kind: Service
      name: resnet-classifier
```

```python
# event-receiver.py — handle CloudEvents
from cloudevents.http import from_http
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["POST"])
def handle_event():
    event = from_http(request.headers, request.get_data())
    if event["type"] == "image.uploaded":
        image_url = event["data"]["url"]
        result = model.predict(image_url)
        # emit result event
    return "", 204
```

## Cold Start Optimization

```yaml
# Keep 1 pod always warm for latency-sensitive models
metadata:
  annotations:
    autoscaling.knative.dev/minScale: "1"
```
```yaml
# Queue proxy resource tuning for faster cold starts
spec:
  template:
    metadata:
      annotations:
        queue.sidecar.serving.knative.dev/resource-requests: "100m"
```

## Best Practices

- Set `minScale: 0` for cost savings on non-critical models; `minScale: 1` for latency-sensitive ones
- Use concurrency-based autoscaling for CPU-bound models, RPS-based for I/O-bound
- Pin revisions via `traffic` to lock model versions before validation gates
- Use `activator` mode for scale-from-zero (default); configure `timeoutSeconds` for long inference
- Tag revisions (e.g., `--tag staging`) for environment-specific routing without traffic
- Monitor Knative metrics (activator requests per second, revision concurrency) with Prometheus

## Resources

- [Knative Serving Docs](https://knative.dev/docs/serving/)
- [Knative Eventing Docs](https://knative.dev/docs/eventing/)
- [Autoscaling Guide](https://knative.dev/docs/serving/autoscaling/)
- [Samples](https://knative.dev/docs/serving/samples/)
