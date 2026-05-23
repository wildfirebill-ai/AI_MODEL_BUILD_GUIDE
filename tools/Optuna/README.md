# Optuna

**Version:** 3.6+

**Purpose:** Bayesian hyperparameter optimization framework for automated hyperparameter search. Optuna uses TPE (Tree-structured Parzen Estimator) and CMA-ES samplers to explore hyperparameter spaces efficiently, with built-in pruning to terminate unpromising trials early. Essential for finding optimal learning rates, batch sizes, weight decays, and model architecture parameters.

## Installation

```powershell
pip install optuna
```

Pin version:
```powershell
pip install "optuna>=3.6,<4"
```

Optional dependencies for visualization and distributed optimization:
```powershell
pip install optuna[dashboard]  # Web dashboard
pip install optuna[integration]  # PyTorch Lightning, TensorBoard integration
```

## Basic Usage

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
    warmup = trial.suggest_int("warmup_steps", 100, 2000)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    model = create_model(lr=lr, weight_decay=wd, dropout=dropout)
    val_loss = train_and_eval(model, batch_size=batch_size, warmup=warmup)
    return val_loss

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=50, timeout=7200)  # 50 trials or 2 hours

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

## Advanced Usage / Configuration

### Study persistence and visualization
```python
# Persist to SQLite
study = optuna.create_study(storage="sqlite:///optuna.db", study_name="wfb-sweep", load_if_exists=True)

# Web dashboard
# $ optuna-dashboard sqlite:///optuna.db

# Plot results
from optuna.visualization import plot_parallel_coordinate, plot_contour, plot_optimization_history
fig = plot_parallel_coordinate(study)
fig.show()
```

### Pruning (early stopping)
```python
# Report intermediate values and trigger pruning
for epoch in range(num_epochs):
    val_loss = train_one_epoch(model)
    trial.report(val_loss, step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

### Distributed optimization
```bash
# Run multiple workers connected to the same storage
optuna study optimize objective.py --study-name wfb-sweep --storage sqlite:///optuna.db --n-trials 100
```

## Integration with WFB Model
Optuna is used in the WFB model project for:
- `scripts/hparam_sweep.py` — automated hyperparameter search for training runs
- `wfb_model/tuning/` — trial wrapper functions that report intermediate losses for pruning
- Each trial runs a shortened training loop (e.g., 20% of full epochs) and reports validation perplexity
- Best hyperparameters are saved to `configs/best_hparams.yaml`

## Common Pitfalls / Troubleshooting
- **Trials not reproducible:** Set `sampler=optuna.samplers.TPESampler(seed=42)` for deterministic results
- **Pruning too aggressive:** Adjust `MedianPruner(n_startup_trials=5, n_warmup_steps=3)` — increase `n_warmup_steps` to let trials stabilize before pruning
- **SQLite database locked:** Multiple processes writing to the same SQLite file causes conflicts — use PostgreSQL for large-scale distributed sweeps, or set `storage` to a unique file per process
- **"Trial timed out" but GPU idle:** Increase `timeout` or reduce `n_trials`; ensure the objective function isn't blocking on data loading
- **Memory leak across trials:** Use `optuna.study.Study.stop()` between trials; call `torch.cuda.empty_cache()` at the end of each trial
- **Incompatible with multiprocessing dataloader:** Optuna processes may spawn threads — use `if __name__ == "__main__":` guard in the objective

## Documentation
- https://optuna.org/
- https://optuna.readthedocs.io/
