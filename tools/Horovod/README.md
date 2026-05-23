# Horovod — Distributed Deep Learning Training

Distributed training framework using efficient allreduce (NCCL, MPI, Gloo) across multi-GPU and multi-node.

## Key Components

| Component | Description |
|-----------|-------------|
| `horovod.torch` | PyTorch integration |
| `hvd.DistributedOptimizer` | Wraps any optimizer for distributed training |
| `hvd.broadcast_parameters()` | Broadcast params from rank 0 |
| `hvd.allreduce()` | Gradient averaging across workers |
| Compression | FP16, 1-bit SGD to reduce communication |

## Installation

```bash
pip install horovod
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[tensorflow]
```

## PyTorch — Multi-GPU Training

```python
import torch, torch.nn as nn, torch.optim as optim
import horovod.torch as hvd
from torch.utils.data import DataLoader, TensorDataset

hvd.init()
torch.cuda.set_device(hvd.local_rank())

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).cuda()
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

optimizer = hvd.DistributedOptimizer(
    optim.SGD(model.parameters(), lr=0.01 * hvd.size()),
    named_parameters=model.named_parameters(),
)

dataset = TensorDataset(torch.randn(1000, 1, 28, 28), torch.randint(0, 10, (1000,)))
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=hvd.size(), rank=hvd.rank())
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

loss_fn = nn.CrossEntropyLoss()
for epoch in range(5):
    model.train()
    sampler.set_epoch(epoch)
    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
    if hvd.rank() == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if hvd.rank() == 0:
    torch.save(model.state_dict(), "model.pt")
```

## Lightning, Allreduce & Inference

```python
import pytorch_lightning as pl
from pytorch_lightning.strategies import HorovodStrategy

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    def training_step(self, batch, batch_idx):
        return nn.CrossEntropyLoss()(self.net(batch[0]), batch[1])
    def configure_optimizers(self):
        return optim.SGD(self.net.parameters(), lr=0.01)

pl.Trainer(accelerator="gpu", devices=4, strategy=HorovodStrategy(), max_epochs=10).fit(LitModel(), loader)

# Manual allreduce
model = nn.Linear(10, 2).cuda()
for _ in range(5):
    loss_fn(model(X.cuda()), y.cuda()).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = hvd.allreduce(p.grad.data, average=True)
    optimizer.step(); optimizer.zero_grad()
```

## TensorFlow & Running

```python
import tensorflow as tf, horovod.tensorflow as hvd
hvd.init()
tf.config.experimental.set_memory_growth(
    tf.config.experimental.list_physical_devices('GPU')[hvd.local_rank()], True)
model = tf.keras.Sequential([tf.keras.layers.Dense(128, 'relu'),
                              tf.keras.layers.Dense(10, 'softmax')])
optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.SGD(0.01 * hvd.size()),
                                      compression=hvd.Compression.fp16)
hvd.broadcast_variables(model.variables, root_rank=0)
hvd.broadcast_variables(optimizer.variables(), root_rank=0)
```

```bash
horovodrun -np 4 python train.py                    # single node, 4 GPUs
horovodrun -np 8 -H host1:4,host2:4 python train.py # multi-node
mpirun -np 4 -x NCCL_DEBUG=INFO python train.py     # via MPI
```

## Resources

- [Horovod Docs](https://horovod.readthedocs.io/)
- [GitHub](https://github.com/horovod/horovod)
- [PyTorch Guide](https://horovod.readthedocs.io/en/stable/pytorch.html)
