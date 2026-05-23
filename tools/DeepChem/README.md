# DeepChem — Deep Learning for Drug Discovery

**Version:** 2.8 / `deepchem>=2.8.0`

## Purpose

DeepChem provides deep learning tools for drug discovery, computational chemistry, and molecular science. It offers domain-specific featurizers (graphs, fingerprints, grids), model architectures (graph convolutions, transformers, GANs for molecules), and dataset handling for cheminformatics. DeepChem integrates with RDKit, PyTorch, TensorFlow, and JAX backends.

## Installation

```bash
# Requires RDKit (conda recommended)
conda install -c conda-forge rdkit
pip install "deepchem[torch,rdkit]==2.8.0"

# For full support (TensorFlow too):
pip install "deepchem[all]==2.8.0"
```

## Basic Usage Example

```python
import deepchem as dc

# Load the FreeSolv dataset (hydration free energy)
tasks, datasets, transformers = dc.molnet.load_freesolv(featurizer="GraphConv")
train, valid, test = datasets

# Build a GraphConvModel
model = dc.models.GraphConvModel(
    n_tasks=len(tasks),
    batch_size=32,
    mode="regression",
    dropout=0.2,
)

# Train
model.fit(train, nb_epoch=50)

# Evaluate
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
scores = model.evaluate(test, [metric], transformers)
print(f"Test R²: {scores['pearson_r2_score']:.3f}")

# Predict
predictions = model.predict(test)
```

### Custom dataset from molecular SMILES

```python
smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
properties = [0.0, 1.0, 0.5]

featurizer = dc.feat.CircularFingerprint(size=2048)
features = featurizer.featurize(smiles)
dataset = dc.data.NumpyDataset(X=features, y=properties)
```

## Advanced Usage / Configuration

- **Molecular featurization**: `GraphConv` (graphs with atom/bond features), `Weave` (pairwise), `ECFP` (circular fingerprints), `RDKitDescriptors` (200+ descriptors), `CoulombMatrix` (for quantum properties).
- **Model zoo**: `GraphConvModel`, `AttentiveFPModel`, `MPNNModel`, `DAGModel`, `GATModel`, `WeaveModel`, `TextCNNModel` (for protein sequences), `SeqToSeq` (molecule generation).
- **Hyperparameter search**: `dc.hyper.GridHyperparamOpt()` or `dc.hyper.RandomHyperparamOpt()` with K-fold cross-validation.
- **Uncertainty**: `dc.models.EnsembleModel()` wraps multiple estimators; use `dc.metrics.Metric(dc.metrics.mae_score, uncertainty=True)`.
- **Molecular generation**: `dc.models.MolecularGAN` and `dc.models.ChemBERTa` for generative tasks.
- **Splitters**: `dc.splits.ScaffoldSplitter`, `ButinaSplitter`, `RandomSplitter`, `TimeSplitter` — essential for realistic generalization estimates.
- **Reinforcement learning**: `dc.models.SE3Transformer` for equivariant 3D predictions and `dc.dock.pose_scoring` for docking.

## Integration with the WFB Model Project

```python
# wfb_model/deepchem/tox21_bench.py
import deepchem as dc

def train_wfb_tox21():
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer="GraphConv")
    train, valid, test = datasets
    model = dc.models.GraphConvModel(n_tasks=len(tasks), mode="classification")
    model.fit(train, nb_epoch=30)
    return model

# WFB wraps DeepChem models in a standardized Trainer class:
# wfb_model/trainers/deepchem_trainer.py
```

WFB uses DeepChem for ADMET prediction, molecular property regression, and drug-target interaction modeling. All molecule data is stored as SMILES in `wfb_model/data/chem/`. Preprocessing pipelines convert raw `.sdf` or `.csv` files into `dc.data.Dataset` objects.

## Common Pitfalls / Troubleshooting

- **RDKit import errors**: RDKit must be installed *before* DeepChem. Use `conda install -c conda-forge rdkit` rather than pip.
- **GraphConv OOM on large molecules**: Increase `batch_size` slowly or reduce `max_atoms` in featurizer (e.g., `GraphConv(mode="fc"` sets a cutoff).
- **Featurization fails on invalid SMILES**: Use `dc.utils.is_valid_smiles()` to filter rows before featurizing; RDKit might segfault on malformed SMILES.
- **Dataset memory**: `dc.data.Dataset` loads into memory. For >100K molecules use `dc.data.DiskDataset` and batch generators.
- **Backend mismatch**: Ensure `dc.models` imports the correct backend — set `DEEPCHEM_BACKEND=torch` (default), `tensorflow`, or `jax` before import.
- **Weave featurizer is slow**: It computes pairwise distances for all atom pairs (O(n²)). Subsample molecules to max 50 atoms if possible.

## Documentation Links

- DeepChem docs: https://deepchem.readthedocs.io/
- GitHub: https://github.com/deepchem/deepchem
- Model reference: https://deepchem.readthedocs.io/en/latest/api_reference/models.html
- Featurizers: https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html
- Tutorials: https://github.com/deepchem/deepchem/tree/master/examples/tutorials
