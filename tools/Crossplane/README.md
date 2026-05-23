# Crossplane — Kubernetes Control Plane Framework

Crossplane lets platform teams compose cloud infrastructure into custom Kubernetes resources, managed declaratively via `kubectl`. It turns any cloud API into a CRD.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Provider** | A controller that reconciles external resources (AWS, GCP, Azure, Helm, Kubernetes) |
| **Managed Resource (MR)** | A CRD instance representing a single cloud resource (e.g., `RDSInstance`, `Bucket`) |
| **Composition** | A template that composes multiple managed resources into a single higher-level abstraction |
| **Composite Resource (XR)** | An instance of a Composition — the user-facing custom resource |
| **Claim (XRC)** | A namespaced Composite Resource — allows tenants to provision infra without cluster-wide access |
| **XRDs (CompositeResourceDefinition)** | The schema for a Composite Resource (analogous to CRD definitions) |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Kubernetes Cluster                  │
│                                                       │
│  ┌──────────────┐     ┌──────────────────────────┐   │
│  │  User/Claim   │────▶│   Composite Resource (XR)│   │
│  │  (namespace)  │     │   ┌──────────────────┐  │   │
│  └──────────────┘     │   │   Composition      │  │   │
│                       │   │   ┌──────────────┐ │  │   │
│                       │   │   │ Managed Res 1 │ │  │   │
│                       │   │   │ Managed Res 2 │ │  │   │
│                       │   │   └──────────────┘ │  │   │
│                       │   └──────────────────┘  │   │
│                       └──────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## Installation

```bash
helm repo add crossplane-stable https://charts.crossplane.io/stable
helm install crossplane crossplane-stable/crossplane --namespace crossplane-system --create-namespace

# Install a provider
kubectl crossplane install provider crossplane/provider-helm:master
kubectl crossplane install provider crossplane/provider-kubernetes:master
kubectl crossplane install provider crossplane/provider-gcp:master
```

## Integration: Managed ML Infrastructure with CRDs

Define a custom `MLWorkspace` resource that provisions a complete ML environment: K8s namespace, model service, Postgres DB, S3 bucket, and IAM role.

### XRD — Define the Composite Schema

```yaml
apiVersion: apiextensions.crossplane.io/v1
kind: CompositeResourceDefinition
metadata:
  name: xmlworkspaces.ml.example.org
spec:
  group: ml.example.org
  names:
    kind: XMLWorkspace
    plural: xmlworkspaces
  claimNames:
    kind: MLWorkspace
    plural: mlworkspaces
  versions:
    - name: v1alpha1
      served: true
      referenceable: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                model:
                  type: string
                replicas:
                  type: integer
                  default: 2
                storageGB:
                  type: integer
                  default: 100
                gpu:
                  type: boolean
                  default: false
              required:
                - model
```

### Composition — Orchestrate Resources

```yaml
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: mlworkspace-standard
  labels:
    provider: kubernetes
    type: standard
spec:
  compositeTypeRef:
    apiVersion: ml.example.org/v1alpha1
    kind: XMLWorkspace
  resources:
    - name: model-service
      base:
        apiVersion: kubernetes.crossplane.io/v1alpha1
        kind: Object
        spec:
          forProvider:
            manifest:
              apiVersion: serving.knative.dev/v1
              kind: Service
              spec:
                template:
                  spec:
                    containers:
                      - image: ml-server:latest
                        ports:
                          - containerPort: 8080
      patches:
        - type: FromCompositeFieldPath
          fromFieldPath: spec.model
          toFieldPath: spec.forProvider.manifest.spec.template.spec.containers[0].image
        - type: FromCompositeFieldPath
          fromFieldPath: spec.replicas
          toFieldPath: spec.forProvider.manifest.spec.template.metadata.annotations["autoscaling.knative.dev/minScale"]

    - name: namespace
      base:
        apiVersion: kubernetes.crossplane.io/v1alpha1
        kind: Object
        spec:
          forProvider:
            manifest:
              apiVersion: v1
              kind: Namespace
              metadata:
                labels:
                  type: ml-workspace

    - name: model-bucket
      base:
        apiVersion: s3.aws.crossplane.io/v1beta1
        kind: Bucket
        spec:
          forProvider:
            locationConstraint: us-east-1
            acl: private
          providerConfigRef:
            name: aws-provider
      patches:
        - type: FromCompositeFieldPath
          fromFieldPath: spec.storageGB
          toFieldPath: spec.forProvider.storageGB

    - name: postgres-db
      base:
        apiVersion: database.gcp.crossplane.io/v1beta1
        kind: CloudSQLInstance
        spec:
          forProvider:
            databaseVersion: POSTGRES_13
            tier: db-custom-2-7680
            region: us-central1
          writeConnectionSecretToRef:
            namespace: crossplane-system
      patches:
        - type: FromCompositeFieldPath
          fromFieldPath: metadata.name
          toFieldPath: spec.writeConnectionSecretToRef.name
```

### Claim — User Creates ML Environment

```yaml
apiVersion: ml.example.org/v1alpha1
kind: MLWorkspace
metadata:
  name: team-alpha-bert
  namespace: ml-teams
spec:
  model: "bert-base-uncased"
  replicas: 3
  storageGB: 200
  gpu: true
  compositionRef:
    name: mlworkspace-standard
```

Apply with:

```bash
kubectl apply -f mlworkspace-claim.yaml

# Watch resources get provisioned
kubectl get managed
kubectl get xmlworkspaces
kubectl get mlworkspaces -n ml-teams

# Delete the claim → all composed resources are garbage collected
kubectl delete mlworkspace team-alpha-bert -n ml-teams
```

### Provider Configuration

```yaml
apiVersion: aws.crossplane.io/v1beta1
kind: ProviderConfig
metadata:
  name: aws-provider
spec:
  credentials:
    source: Secret
    secretRef:
      namespace: crossplane-system
      name: aws-creds
      key: creds
```

```bash
# Create provider secret
kubectl create secret generic aws-creds \
  -n crossplane-system \
  --from-file=creds=./aws-credentials.json
```

## Composition Functions

For complex ML infra logic, use composition functions (e.g., Patch-and-Transform, or custom Go/Python functions):

```yaml
spec:
  functions:
    - name: auto-scale-config
      type: Container
      image: ghcr.io/org/crossplane-fn/auto-scale:latest
      config:
        minReplicas: 1
        maxReplicas: 20
    - name: tag-resources
      type: Patch
      patches:
        - type: FromCompositeFieldPath
          fromFieldPath: metadata.labels
          toFieldPath: spec.forProvider.tags
```

## Best Practices

- Define XRDs for platform abstractions (MLWorkspace, ModelEndpoint, FeatureStore); keep claims namespaced
- Use Composition `patchSets` to reuse common patches (e.g., region, tags, provider config)
- Enable `deletionPolicy: Delete` on managed resources for clean teardown
- Use `provider-helm` to deploy Helm charts as part of composite resources
- Store connection secrets (DB passwords, bucket keys) in Vault via `provider-kubernetes`
- Set resource limits on Crossplane providers to avoid OOM in large environments

## Resources

- [Crossplane Docs](https://docs.crossplane.io/)
- [Composition Guide](https://docs.crossplane.io/latest/concepts/composition/)
- [Provider Catalog](https://github.com/crossplane-contrib)
- [Upbound Marketplace](https://marketplace.upbound.io/)
