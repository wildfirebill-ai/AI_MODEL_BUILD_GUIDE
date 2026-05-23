# scikit-learn

**Version:** 1.5+

**Purpose:** General-purpose machine learning library for data preprocessing, clustering, dimensionality reduction, and evaluation metrics. While not used for training the WFB model itself, scikit-learn provides essential utilities for data analysis, embedding quality evaluation, data poisoning detection, and annotation agreement scoring in the data pipeline.

## Installation

```powershell
pip install scikit-learn
```

Pin version:
```powershell
pip install "scikit-learn>=1.5,<1.6"
```

### Platform notes
- **Windows:** Pre-built wheels available for all Python versions — direct `pip install` works
- **Apple Silicon (M1/M2/M3):** Native wheels available for 1.5+ — no Rosetta needed
- **Linux:** CUDA not used (scikit-learn is CPU-only) — no GPU requirements

## Basic Usage in LLM Workflows

```python
# DBSCAN clustering for data poisoning / outlier detection
from sklearn.cluster import DBSCAN
import numpy as np

embeddings = np.random.randn(1000, 768)  # 1000 documents
clustering = DBSCAN(eps=0.3, min_samples=5).fit(embeddings)
n_noise = list(clustering.labels_).count(-1)
print(f"Outliers detected: {n_noise}")

# K-means for diversity sampling in active learning
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=100, random_state=42).fit(embeddings)
# Select one sample per cluster for labeling
selected = [np.where(kmeans.labels_ == i)[0][0] for i in range(100)]

# Cohen's kappa for inter-annotator agreement
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotator_a_labels, annotator_b_labels)
print(f"Annotator agreement: {kappa:.3f}")

# PCA for embedding visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2).fit_transform(embeddings)
plt.scatter(pca[:, 0], pca[:, 1], c=cluster_labels, alpha=0.5)
plt.savefig("embeddings_viz.png")
```

## Advanced Usage / Configuration

### Key modules for LLM workflows
| Module | Use Case | Example |
|--------|----------|---------|
| `sklearn.cluster` | DBSCAN, KMeans, HDBSCAN | Data dedup, diversity sampling |
| `sklearn.metrics` | Accuracy, kappa, precision/recall | Evaluation, annotation quality |
| `sklearn.decomposition` | PCA, TruncatedSVD | Embedding visualization |
| `sklearn.feature_extraction` | CountVectorizer, TfidfVectorizer | Baseline text features |
| `sklearn.model_selection` | train_test_split, StratifiedKFold | Dataset splitting |
| `sklearn.preprocessing` | StandardScaler, normalize | Embedding normalization |
| `sklearn.manifold` | TSNE, Isomap | Embedding visualization (slower) |

### Cache and parallelism
```python
# Enable joblib caching for repeated operations
from sklearn.externals import joblib
joblib.dump(kmeans, "kmeans_model.pkl")

# Control parallelism
from sklearn.cluster import DBSCAN
DBSCAN(n_jobs=-1)  # Use all CPU cores
```

## Integration with WFB Model
scikit-learn is used in the WFB model project for:
- `scripts/data_dedup.py` — DBSCAN-based embedding deduplication for training data cleaning
- `scripts/annotation_agreement.py` — Cohen's kappa and Krippendorff's alpha for quality control
- `scripts/eval_diversity.py` — embedding diversity metrics using silhouette score and clustering
- `wfb_model/data/sampling.py` — K-means-based active learning sampling for efficient labeling
- Data preprocessing pipelines that use `train_test_split` and cross-validation

## Common Pitfalls / Troubleshooting
- **`ValueError: could not convert float to int`:** scikit-learn requires numerical data — ensure all feature columns are numeric; use `sklearn.preprocessing.LabelEncoder` for categorical variables
- **DBSCAN memory with large datasets:** DBSCAN builds a pairwise distance matrix — O(n²) memory; for >100K points, use `HDBSCAN` (approximate) or subsample
- **K-means initialization sensitivity:** Results vary with different `random_state` values — run multiple initializations (`n_init=10`) or use `k-means++`
- **PCA with sparse embeddings:** Use `TruncatedSVD` instead of PCA for sparse matrices (e.g., TF-IDF vectors)
- **`train_test_split` leaking information:** Ensure splits are done before any preprocessing that uses global statistics (fit `StandardScaler` on training set only)
- **`n_jobs=-1` causing system freeze:** scikit-learn uses CPU multiprocessing — on very large systems, cap with `n_jobs=psutil.cpu_count(logical=False) // 2`

## Documentation
- https://scikit-learn.org/stable/
- https://scikit-learn.org/stable/user_guide.html
