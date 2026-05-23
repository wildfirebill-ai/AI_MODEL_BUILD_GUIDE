# Harbor — Cloud-Native Container Registry

Harbor is an open-source container image registry with built-in security, identity, and replication. It stores and manages OCI artifacts (Docker images, Helm charts, OCI artifacts) with vulnerability scanning, RBAC, and geo-replication.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Project** | Logical grouping of repositories with configurable access policies (public/private) |
| **Replication** | Push/pull-based synchronization of artifacts between Harbor instances or registries |
| **Vulnerability Scanning** | Automatic CVE scanning via Trivy, Clair, or Aqua (on push or scheduled) |
| **RBAC** | Role-based access (admin, maintainer, developer, guest, custom roles) per project |
| **Artifact Types** | OCI-compliant: Docker images, Helm charts, CNAB, OCI artifacts (model files, datasets) |
| **Garbage Collection** | Blob cleanup for untagged and unreferenced artifacts to reclaim storage |
| **Notary** | Content trust and image signing using Docker Content Trust / Cosign |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Harbor                             │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐  │
│  │ Core     │ │ Registry  │ │ Scanner│ │ Replication│  │
│  │ (API/UI)│ │ (Docker) │ │ (Trivy)  │  │ Service  │  │
│  └────┬────┘ └────┬─────┘ └────────┘ └──────┬─────┘  │
│       │           │                         │        │
│  ┌────┴───────────┴─────────────────────────┴──────┐ │
│  │              Storage (S3/GCS/Azure/FS)           │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
# Helm install
helm repo add harbor https://helm.goharbor.io
helm install harbor harbor/harbor --namespace harbor --create-namespace \
  --set expose.type=loadBalancer \
  --set persistence.enabled=true \
  --set trivy.enabled=true \
  --set chartmuseum.enabled=true \
  --set notary.enabled=true

# Get URL and admin password
kubectl get svc -n harbor
kubectl get secret harbor-harbor-core -n harbor -o jsonpath="{.data.HARBOR_ADMIN_PASSWORD}" | base64 -d
```

## Integration: Container Image Storage for ML Models

Harbor is ideal for storing model container images (TorchServe, Triton, BentoML) and OCI artifacts (model weights, datasets).

### Push a Model Image

```yaml
# Dockerfile for model serving image
FROM pytorch/torchserve:0.9.0-gpu

USER root
COPY model-store/ /home/model-server/model-store/
COPY config.properties /home/model-server/config.properties
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

USER model-server

EXPOSE 8080 8081
CMD ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models", "bert=bert.mar"]
```

```bash
# Login to Harbor
docker login harbor.ml.example.com --username admin

# Build and tag
docker build -t model-serving/bert:v2.1.0 .
docker tag model-serving/bert:v2.1.0 harbor.ml.example.com/ml-models/bert:v2.1.0

# Push
docker push harbor.ml.example.com/ml-models/bert:v2.1.0

# Push another tag
docker tag model-serving/bert:v2.1.0 harbor.ml.example.com/ml-models/bert:latest
docker push harbor.ml.example.com/ml-models/bert:latest
```

### Store Model Weights as OCI Artifacts

```bash
# Install OCI CLI tool (oras)
# Download from https://github.com/oras-project/oras/releases

# Push model weights as an OCI artifact
oras push harbor.ml.example.com/ml-models/bert-weights:v2.1.0 \
  ./pytorch_model.bin:application/vnd.ml.model.weights \
  ./config.json:application/vnd.ml.model.config \
  ./vocab.txt:text/plain

# Pull and use
oras pull harbor.ml.example.com/ml-models/bert-weights:v2.1.0
```

### Python: Push/Pull from Harbor

```python
import docker
import requests

# Docker SDK
client = docker.from_env()
client.login("harbor.ml.example.com", username="robot$ml-builder", password="xxx")

# Build and push
image, logs = client.images.build(path="./", tag="torchserve:latest")
client.images.push("harbor.ml.example.com/ml-models/torchserve:latest")

# Harbor API: list repositories
resp = requests.get(
    "https://harbor.ml.example.com/api/v2.0/projects/ml-models/repositories",
    auth=("admin", "password"),
)
repos = resp.json()
for repo in repos:
    print(repo["name"], repo["artifact_count"])
```

## Vulnerability Scanning

```yaml
# Automatically scan all artifacts in a project
apiVersion: goharbor.io/v1alpha1
kind: Project
metadata:
  name: ml-models
spec:
  ---
# Via Harbor UI: Project → Configuration → Automatically scan images on push
```
```bash
# Manual scan via API
curl -u admin:password -X POST \
  "https://harbor.ml.example.com/api/v2.0/projects/ml-models/repositories/bert/artifacts/v2.1.0/scan"

# Get scan results
curl -u admin:password \
  "https://harbor.ml.example.com/api/v2.0/projects/ml-models/repositories/bert/artifacts/v2.1.0/additions/vulnerabilities"

# Filter critical vulnerabilities
curl -u admin:password -s \
  "https://harbor.ml.example.com/api/v2.0/projects/ml-models/repositories/bert/artifacts/v2.1.0/additions/vulnerabilities" \
  | jq '.[] | select(.severity == "Critical")'
```

### CI/CD Scan Gate

```yaml
# GitLab CI — block deploy if Critical CVEs found
harbor-scan:
  stage: test
  script:
    - apk add curl jq
    - |
      # Trigger scan
      curl -u "admin:$HARBOR_PASSWORD" -X POST \
        "https://harbor.ml.example.com/api/v2.0/projects/ml-models/repositories/$CI_COMMIT_REF_NAME/artifacts/$CI_COMMIT_SHA/scan"
      sleep 10
      # Check results
      CRITICAL=$(curl -u "admin:$HARBOR_PASSWORD" -s \
        "https://harbor.ml.example.com/api/v2.0/projects/ml-models/repositories/$CI_COMMIT_REF_NAME/artifacts/$CI_COMMIT_SHA/additions/vulnerabilities" \
        | jq '[.[] | select(.severity == "Critical")] | length')
      if [ "$CRITICAL" -gt "0" ]; then
        echo "Blocking: $CRITICAL Critical CVEs found"
        exit 1
      fi
```

## Replication Across Regions

```yaml
# replication-rule.yaml — Harbor API
POST /api/v2.0/replication/policies
{
  "name": "pull-through-mirror",
  "description": "Mirror public models to local Harbor",
  "src_registry": {
    "type": "docker-hub",
    "endpoint": "https://hub.docker.com",
    "credential": {
      "access_key": "username",
      "access_secret": "password"
    }
  },
  "dest_namespace": "ml-models/mirror",
  "filters": [
    {
      "type": "name",
      "value": "pytorch/torchserve"
    }
  ],
  "trigger": {
    "type": "event_based"
  },
  "deletion": true,
  "override": true
}
```

## Robot Accounts for CI/CD

```bash
# Create a robot account via API
curl -u admin:password -X POST \
  "https://harbor.ml.example.com/api/v2.0/robots" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "github-actions",
    "duration": -1,
    "level": "project",
    "permissions": [
      {
        "kind": "project",
        "namespace": "ml-models",
        "access": [
          {"resource": "repository", "action": "push"},
          {"resource": "repository", "action": "pull"}
        ]
      }
    ]
  }'

# Use in GitHub Actions
# docker login -u 'robot$github-actions' -p <token> harbor.ml.example.com
```

## Garbage Collection

```bash
# Create GC schedule via API
curl -u admin:password -X POST \
  "https://harbor.ml.example.com/api/v2.0/system/gc/schedule" \
  -H "Content-Type: application/json" \
  -d '{
    "schedule": {
      "type": "Weekly",
      "weekday": 0,
      "offtime": 0
    },
    "parameters": {
      "delete_untagged": true,
      "dry_run": false
    }
  }'

# Manual GC
curl -u admin:password -X POST \
  "https://harbor.ml.example.com/api/v2.0/system/gc"
```

## Image Signing with Cosign

```bash
# Generate key pair
cosign generate-key-pair

# Sign the image
cosign sign --key cosign.key \
  harbor.ml.example.com/ml-models/bert:v2.1.0

# Verify before deployment
cosign verify --key cosign.pub \
  harbor.ml.example.com/ml-models/bert:v2.1.0

# In Kubernetes: deploy only signed images
kubectl create secret generic cosign-pubkey --from-file=cosign.pub
```

## Best Practices

- Use robot accounts for CI/CD with minimal permissions (push only to specific projects)
- Enable vulnerability scanning on push with Trivy and set severity thresholds in CI
- Tag images with semantic versions (`v2.1.0`), not just `latest`, for reproducible deployments
- Set up replication rules to mirror public base images (e.g., `pytorch/pytorch`) for air-gapped environments
- Schedule weekly garbage collection to clean up untagged and dangling blobs
- Enable content trust (Notary/Cosign) for production model images
- Use Harbor's project quotas to prevent storage exhaustion by team

## Resources

- [Harbor Docs](https://goharbor.io/docs/)
- [Harbor Helm Chart](https://github.com/goharbor/harbor-helm)
- [Harbor API v2.0](https://goharbor.io/docs/2.10.0/build-customize-contribute/configure-harbor-api/)
- [Cosign + Harbor](https://goharbor.io/docs/2.10.0/working-with-projects/working-with-images/sign-images/)
