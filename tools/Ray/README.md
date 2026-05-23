# Ray

**Version:** 2.40+

**Purpose:** Unified distributed computing framework for scaling ML workloads from a laptop to a cluster. Ray provides distributed task scheduling (`@ray.remote`), data processing (Ray Data), hyperparameter tuning (Ray Tune), model training (Ray Train), and model serving (Ray Serve) under a single API. For the WFB model pipeline, Ray orchestrates distributed training across multiple GPUs, parallelizes data preprocessing, and serves the trained model in production.

## Installation

```powershell
pip install "ray[default,train,tune,serve]==2.40.0"
```

## Basic Usage

### Distributed Tasks

```python
import ray

ray.init()

@ray.remote(num_gpus=1)
def train_shard(config: dict) -> dict:
    # Train a model shard on one GPU
    return {"loss": 0.02, "accuracy": 0.97}

futures = [train_shard.remote({"lr": 1e-4}) for _ in range(4)]
results = ray.get(futures)
print(results)
```

### Ray Train (Distributed Training)

```python
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func(config):
    import torch
    model = torch.nn.Linear(1024, 1024).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    for epoch in range(3):
        loss = model(torch.randn(32, 1024).cuda()).mean()
        loss.backward()
        optimizer.step()
        train.report({"loss": loss.item()})

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    train_loop_config={"lr": 3e-4},
)
result = trainer.fit()
print(result.metrics)
```

## Advanced Usage

### Ray Serve for Model Deployment

```python
from ray import serve
from starlette.requests import Request

@serve.deployment(ray_actor_options={"num_gpus": 1})
class WFBModel:
    def __init__(self):
        import torch
        self.model = torch.nn.Linear(1024, 1024).cuda()
        self.model.eval()

    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        import torch
        x = torch.tensor(data["input"]).cuda()
        with torch.no_grad():
            out = self.model(x)
        return {"output": out.tolist()}

serve.run(WFBModel.bind())
```

### Hyperparameter Tuning with Ray Tune

```python
from ray import tune
from ray.tune.search.optuna import OptunaSearch

tuner = tune.Tuner(
    train_func,
    tune_config=tune.TuneConfig(
        num_samples=20,
        search_alg=OptunaSearch(),
        metric="loss",
        mode="min",
    ),
    param_space={"lr": tune.loguniform(1e-5, 1e-3)},
)
results = tuner.fit()
best_config = results.get_best_result().config
```

## Integration with WFB Model

In the WFB pipeline, Ray serves as the distributed orchestration layer during **Parts 5–8** (Pretraining, Post-Training, Optimization, Production). Use Ray Train with PyTorch FSDP or DeepSpeed to scale pretraining across nodes. Ray Tune automates hyperparameter sweeps for learning rate, weight decay, and scheduler parameters. After quantization (AWQ/GPTQ), Ray Serve deploys the model behind a REST endpoint with auto-scaling, canarying, and request batching — replacing raw FastAPI for production traffic.

## Common Pitfalls

- **Object store memory**: Ray's shared memory object store fills up with large tensors. Set `object_store_memory` in `ray.init()` or restart with `ray stop`.
- **Serialization**: Custom classes and lambdas fail across workers. Use `ray.cloudpickle` or refactor with plain functions.
- **GPU conflicts**: `num_gpus=0.5` enables GPU sharing but can cause OOM. Prefer whole-GPU assignment for training.
- **Head node restart**: Use `ray.init(address="auto", ignore_reinit_error=True)` to attach to an existing cluster.
- **Logging**: Worker logs are not printed by default. Use `train.report()` or `ray.get` to surface results.

## Documentation

- https://docs.ray.io/
- https://docs.ray.io/en/latest/tune/index.html
- https://docs.ray.io/en/latest/serve/index.html
