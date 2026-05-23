# Helm — Kubernetes Package Manager

Helm is the standard package manager for Kubernetes. It uses **charts** (packaged templates) to define, install, and upgrade complex K8s applications with a single command.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Chart** | A packaged K8s template (deployments, services, configmaps, etc.) |
| **Values** | YAML configuration injected into chart templates at install/upgrade time |
| **Release** | A running instance of a chart on a cluster |
| **Repository** | A collection of published charts (e.g., stable, bitnami) |
| **Hook** | A job that runs at specific points during release lifecycle (pre/post install, upgrade, delete) |

## Core Commands

```bash
# Add a repository
helm repo add bitnami https://charts.bitnami.com/bitnami

# Search for a chart
helm search repo bitnami/nginx

# Install a chart with custom values
helm install my-release bitnami/nginx --values values.yaml

# Dry-run and render templates locally
helm template my-release bitnami/nginx --values values.yaml

# Upgrade a release (rolls out changes)
helm upgrade my-release bitnami/nginx --values values-prod.yaml

# Rollback to a previous revision
helm rollback my-release 1

# List all releases
helm list --all-namespaces
```

## Chart Structure

```
mychart/
├── Chart.yaml          # Metadata (name, version, dependencies)
├── values.yaml         # Default configuration values
├── templates/          # Go-templated K8s manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── _helpers.tpl    # Named template partials
└── charts/             # Subchart dependencies (packed .tgz)
```

## Integration: ML Model Serving on K8s

Helm is ideal for packaging ML serving stacks (e.g., TorchServe, Triton, BentoML, Seldon).

### Example: Deploy a Hugging Face Model with TorchServe

Create `values.yaml`:

```yaml
image:
  repository: pytorch/torchserve
  tag: 0.9.0
  pullPolicy: IfNotPresent

model:
  name: "bert-base-uncased"
  url: "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
  handler: "transformers"
  workers: 2
  batchSize: 4

resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

service:
  type: ClusterIP
  port: 8080
```

Package and deploy:

```bash
# Package the chart
helm package ./torchserve-chart

# Deploy with environment-specific values
helm upgrade --install bert-serving ./torchserve-chart \
  --values values-prod.yaml \
  --namespace ml-serve \
  --create-namespace

# Render templates for CI/CD review
helm template bert-serving ./torchserve-chart \
  --values values-staging.yaml \
  --output-dir ./rendered-manifests
```

### Dependency Management for ML Stack

```yaml
# Chart.yaml
dependencies:
  - name: redis
    version: "~17.0.0"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
  - name: kafka
    version: "~22.0.0"
    repository: "https://charts.bitnami.com/bitnami"
    condition: kafka.enabled
  - name: prometheus
    version: "~15.0.0"
    repository: "https://prometheus-community.github.io/helm-charts"
    condition: monitoring.enabled
```

Update dependencies and redeploy:

```bash
helm dependency update ./ml-platform
helm upgrade --install ml-platform ./ml-platform --namespace ml-infra
```

## Best Practices

- Pin chart versions in production; don't use `latest` tags
- Use `helm template` in CI to validate manifests before deploy
- Split values files by environment (`values-staging.yaml`, `values-prod.yaml`)
- Use `--atomic` flag to auto-rollback on deployment failure
- Store chart repos in OCI-compatible registries (e.g., Harbor, ECR) for versioning
- Leverage library charts for shared templates across ML services

## Advanced: Hooks for Model Validation

```yaml
# templates/validate-model-hook.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ .Release.Name }}-validate"
  annotations:
    "helm.sh/hook": pre-upgrade
    "helm.sh/hook-weight": "-5"
    "helm.sh/hook-delete-policy": hook-succeeded
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: validator
        image: python:3.11-slim
        command: ["python", "-c"]
        args:
          - |
            import requests
            r = requests.post("{{ .Values.validation.endpoint }}",
                              json={"model": "{{ .Values.model.name }}"},
                              headers={"Authorization": "Bearer {{ .Values.validation.token }}"})
            assert r.status_code == 200, f"Model validation failed: {r.text}"
            print("Model validation passed")
```

## Resources

- [Official Docs](https://helm.sh/docs/)
- [Chart Best Practices Guide](https://helm.sh/docs/chart_best_practices/)
- [Artifact Hub](https://artifacthub.io/)
