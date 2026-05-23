# KServe — Serverless Inference on Kubernetes

KServe provides serverless inferencing on Kubernetes with autoscaling, canary rollouts, request batching, and explainability — built on top of Knative and Istio.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **InferenceService** | CRD defining a model deployment |
| **Predictor** | Core component that runs the model |
| **Transformer** | Pre/post-processing step |
| **Explainer** | Provides model explanations (SHAP, LIME) |
| **Knative** | Autoscaling and request-driven serverless |
| **Istio** | Ingress, traffic routing, canary splits |
| **Storage URI** | Model artifact location (S3, GCS, PVC) |

## Basic InferenceService

```yaml
# model.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: s3://models/iris/model.joblib
```

## Deploy

```bash
kubectl apply -f model.yaml

# Check status
kubectl get inferenceservice sklearn-iris
kubectl get pods -l serving.kserve.io/inferenceservice=sklearn-iris
```

## Custom Model Server (Python)

```python
# model_server.py
from kserve import Model, ModelServer

class MyModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.model = load_model("/mnt/models")

    def predict(self, request: dict) -> dict:
        instances = request["instances"]
        predictions = self.model.predict(instances)
        return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    model = MyModel("custom-model")
    ModelServer().start([model])
```

## Canary Deployment (Traffic Split)

```yaml
spec:
  predictor:
    canary:
      sklearn:
        storageUri: s3://models/iris-v2/model.joblib
    canaryTrafficPercent: 10
```

## Autoscaling Configuration

```yaml
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 10  # requests per second per replica
```

## Batching

```yaml
spec:
  predictor:
    batcher:
      maxBatchSize: 32
      maxLatency: 100   # milliseconds
      timeout: 60
```

## Request

```bash
curl -X POST http://<ingress>/v1/models/sklearn-iris:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
```

## Integration Patterns

- **Production Model Serving on K8s**: Serverless scaling with Knative
- **Canary Deployments**: Gradually roll out new model versions
- **A/B Testing**: Split traffic between model variants
- **Explainability**: Attach SHAP/LIME explainer alongside predictor

## References

- [KServe Documentation](https://kserve.github.io/website/)
- [KServe GitHub](https://github.com/kserve/kserve)
- [Knative](https://knative.dev/docs/)
