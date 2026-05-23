# JAX

**Version:** 0.5+

**Purpose:** High-performance numerical computing framework by Google combining NumPy-like API with automatic differentiation, JIT compilation via XLA, and execution on CPU/GPU/TPU. JAX's functional paradigm (`jax.grad`, `jax.jit`, `jax.vmap`, `jax.pmap`) enables research-scale experimentation with explicit hardware control. In the WFB pipeline, JAX is used for prototyping attention variants and TPU-based pretraining via Flax.

## Installation

```powershell
pip install "jax[cuda12]==0.5.0" "flax==0.10.0" "optax==0.2.4"
```

## Basic Usage

### Automatic Differentiation

```python
import jax.numpy as jnp
from jax import random, grad, jit

def loss_fn(w, x, y):
    pred = jnp.dot(x, w)
    return jnp.mean((pred - y) ** 2)

key = random.PRNGKey(0)
w = random.normal(key, (4096,))
x = random.normal(key, (1024, 4096))
y = random.normal(key, (1024,))
grads = grad(loss_fn)(w, x, y)
print(grads.shape)
```

### JIT Compilation

```python
@jit
def ffn(x, w1, w2):
    return jnp.dot(jnp.maximum(jnp.dot(x, w1), 0), w2)

key = random.PRNGKey(0)
result = ffn(
    random.normal(key, (32, 1024)),
    random.normal(key, (1024, 4096)),
    random.normal(key, (4096, 1024)),
)
```

## Advanced Usage

### Flax Model Definition

```python
import flax.linen as nn

class TransformerBlock(nn.Module):
    dim: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.dim
        )
        x = attn(x, x) + x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x) + x
        return nn.LayerNorm()(x)

model = TransformerBlock(dim=512, n_heads=8)
params = model.init(random.PRNGKey(0), jnp.ones((1, 128, 512)))
```

### Data Parallelism with pmap

```python
from jax import pmap

n_devices = 8
params_replicated = pmap(lambda p: p)(params)
batch_replicated = pmap(lambda b: b)(
    jnp.stack([batch] * n_devices)
)
loss, grads = pmap(train_step)(params_replicated, batch_replicated)
```

## Integration with WFB Model

JAX is used in **Part 3** (Architecture Design) and **Part 5** (Pretraining) of the WFB pipeline. For research-heavy phases — novel attention mechanisms, linear attention variants — JAX's `vmap`/`pmap` enable rapid iteration with explicit parallelism control. On TPU (Google Cloud TPU v5e/v5p), JAX/Flax is the recommended pretraining stack. Export trained weights to safetensors for deployment via vLLM or TGI.

## Common Pitfalls

- **Functional purity**: JAX functions must be side-effect-free. In-place ops (`x[i] = 0`) fail silently — use `x.at[i].set(0)`.
- **PRNG state**: Random functions consume the key. Never reuse; split with `random.split(key, n)`.
- **XLA compilation time**: First call is slow. Use `jax.block_until_ready()` to force compilation.
- **TPU memory**: ~8 GB HBM per core. Monitor with `jax.devices()[0].memory_stats()`.
- **Float precision**: Defaults to float32. Enable bfloat16 with `jax.config.update("jax_default_matmul_precision", "bfloat16")`.

## Documentation

- https://jax.readthedocs.io/
- https://flax.readthedocs.io/
- https://optax.readthedocs.io/
