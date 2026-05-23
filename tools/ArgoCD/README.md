# ArgoCD — GitOps Continuous Delivery for Kubernetes

ArgoCD is a declarative GitOps continuous delivery tool for Kubernetes. It synchronises desired application state defined in a Git repository with the live state on the cluster.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Application CRD** | Custom resource defining the source (Git/Helm) and destination cluster |
| **Sync Policy** | Automatic or manual sync strategy (e.g., automated pruning, self-heal) |
| **Health Checks** | Built-in liveness probes for K8s resources; custom Lua scripts supported |
| **SSO** | Single sign-on via Dex, OIDC, Okta, SAML |
| **Multi-Cluster** | Manage applications across many clusters from a single ArgoCD instance |
| **Automated Sync** | Poll repository or use webhook-triggered sync on commit |

## Architecture

```
┌──────────┐     ┌─────────────┐     ┌──────────────┐
│  Git Repo │────▶│  ArgoCD     │────▶│  K8s Cluster  │
│ (manifests)│    │  Controller  │     │  (live state)  │
└──────────┘     └─────────────┘     └──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  ArgoCD Server  │◀──── User / CI
              │  (Web UI + CLI) │
              └────────────────┘
```

## Core CLI Commands

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Get initial admin password
argocd admin initial-password -n argocd

# Login via CLI
argocd login <ARGOCD_SERVER> --sso  # or --username admin

# Create an application from a Git repo
argocd app create bert-serving \
  --repo https://github.com/org/ml-manifests.git \
  --path torchserve/production \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace ml-serve \
  --sync-policy automated \
  --auto-prune \
  --self-heal

# Sync an application manually
argocd app sync bert-serving

# View application status and health
argocd app get bert-serving

# Rollback to a previous deployment
argocd app rollback bert-serving 2

# List all applications across clusters
argocd app list
```

## Application CRD Example

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: model-serving-prod
  namespace: argocd
spec:
  project: ml-models
  source:
    repoURL: https://github.com/org/ml-manifests.git
    targetRevision: main
    path: models/torchserve
    helm:
      valueFiles:
        - values-prod.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-serve
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - ApplyOutOfSyncOnly=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

## Integration: Git-Based Model Deployment with Automated Rollback

Combine ArgoCD with a model evaluation pipeline to rollback when a new model regresses.

### Workflow

```yaml
# ApplicationSet: one app per model version with canary promotion
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: model-versions
  namespace: argocd
spec:
  generators:
    - git:
        repoURL: https://github.com/org/model-registry.git
        revision: HEAD
        directories:
          - path: "models/v*"
  template:
    metadata:
      name: "{{path.basename}}"
    spec:
      source:
        repoURL: https://github.com/org/ml-manifests.git
        targetRevision: "{{path.basename}}"
        path: serving-template
        helm:
          parameters:
            - name: modelVersion
              value: "{{path.basename}}"
      destination:
        namespace: "ml-serve-{{path.basename}}"
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

### Automated Rollback via Webhook + Evaluation

```python
# eval_webhook.py — Trigger rollback if new model regresses
from fastapi import FastAPI, Request
import subprocess
import json

app = FastAPI()

@app.post("/eval-webhook")
async def handle_eval_result(request: Request):
    body = await request.json()
    model_version = body["version"]
    accuracy = body["accuracy"]
    baseline = body["baseline_accuracy"]

    if accuracy < baseline - 0.02:
        # Rollback ArgoCD to previous revision
        subprocess.run([
            "argocd", "app", "rollback",
            f"model-serving-{model_version}", "1"
        ])
        return {"status": "rolled_back", "reason": "accuracy regression"}
    return {"status": "promoted"}
```

## Project Configuration for RBAC

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: ml-models
  namespace: argocd
spec:
  description: "ML Model Serving Applications"
  sourceRepos:
    - "https://github.com/org/ml-manifests.git"
  destinations:
    - namespace: "ml-serve-*"
      server: "https://kubernetes.default.svc"
  clusterResourceWhitelist:
    - group: ""
      kind: Namespace
  roles:
    - name: ml-admin
      policies:
        - p, proj:ml-models:ml-admin, applications, *, ml-models/*, allow
    - name: ml-viewer
      policies:
        - p, proj:ml-models:ml-viewer, applications, get, ml-models/*, allow
```

## Health Check Customization

```yaml
# Use Lua to define custom health for ML services
spec:
  source:
    helm:
      parameters:
        - name: healthScript
          value: |
            hs = {}
            hs.status = "Healthy"
            hs.message = "Model API responding"
            if obj.status.availableReplicas == nil or obj.status.availableReplicas < 1 then
              hs.status = "Degraded"
              hs.message = "No available replicas"
            end
            return hs
```

## Best Practices

- Store Helm values files per environment in the same Git repo
- Enable automated self-heal to recover from drift
- Use ApplicationSets for multi-cluster and multi-tenant deployments
- Configure webhook-based sync (e.g., GitHub webhooks) for faster deployments
- Pin manifests to commit SHAs in production, not branch names
- Use `argocd-notifications` or `argocd-image-updater` for automated image updates

## Resources

- [ArgoCD Docs](https://argo-cd.readthedocs.io/)
- [ApplicationSet Examples](https://argo-cd.readthedocs.io/en/stable/operator-manual/applicationset/)
- [Best Practices](https://argo-cd.readthedocs.io/en/stable/user-guide/best_practices/)
