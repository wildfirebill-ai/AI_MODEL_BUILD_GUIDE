# PyTorch Geometric (PyG) — Graph Neural Networks

Geometric deep learning library on PyTorch. GNN layers, graph data handling, and minibatch sampling.

## Key Components

| Component | Description |
|-----------|-------------|
| `torch_geometric.data.Data` | Graph object with `x`, `edge_index`, `y` |
| `torch_geometric.nn.GCNConv` | Graph Convolutional layer |
| `torch_geometric.nn.GATConv` | Graph Attention layer |
| `torch_geometric.nn.SAGEConv` | GraphSAGE layer |
| `torch_geometric.loader.NeighborLoader` | Minibatch sampling for large graphs |
| `torch_geometric.datasets` | Built-in datasets (Planetoid, TU, etc.) |

## Installation

```bash
pip install torch torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## Data Object

```python
import torch
from torch_geometric.data import Data

x = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
y = torch.tensor([0, 1, 0], dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)
print(data)
```

## GCN — Node Classification on Cora

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

data = Planetoid(root='data/Planetoid', name='Cora')[0]

class GCN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_c, hid_c), GCNConv(hid_c, out_c)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(F.dropout(x, 0.5, self.training), edge_index)

model = GCN(dataset.num_features, 16, dataset.num_classes)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train(); opt.zero_grad()
    loss = F.cross_entropy(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step()
    if epoch % 50 == 0: print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")
pred = model(data.x, data.edge_index).argmax(dim=1)
print(f"Test acc: {(pred[data.test_mask]==data.y[data.test_mask]).sum().item()/data.test_mask.sum().item():.4f}")
## Minibatch & Graph Classification

```python
from torch_geometric.loader import NeighborLoader
loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=256,
                        input_nodes=data.train_mask, shuffle=True)

from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='data/TU', name='ENZYMES')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_c, hid_c), GCNConv(hid_c, hid_c)
        self.lin = torch.nn.Linear(hid_c, out_c)
    def forward(self, x, edge_index, batch):
        x = self.conv2(self.conv1(x, edge_index).relu(), edge_index).relu()
        return self.lin(global_mean_pool(x, batch))

model = GraphClassifier(dataset.num_features, 64, dataset.num_classes)
```

## GAT & Link Prediction

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_c, hid_c, heads)
        self.conv2 = GATConv(hid_c * heads, out_c, 1, concat=False)
    def forward(self, x, edge_index):
        x = F.dropout(x, 0.6, self.training)
        x = self.conv1(x, edge_index).elu()
        x = F.dropout(x, 0.6, self.training)
        return self.conv2(x, edge_index)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_c, hid_c):
        super().__init__()
        self.conv = GCNConv(in_c, hid_c)
        self.lin = torch.nn.Linear(hid_c * 2, 1)
    def forward(self, x, edge_index, edge_label_index):
        src, dst = edge_label_index
        return torch.sigmoid(self.lin(torch.cat([self.conv(x, edge_index)[src],
                                                  self.conv(x, edge_index)[dst]], -1)))
```

## Resources

- [PyG Docs](https://pytorch-geometric.readthedocs.io/)
- [GitHub](https://github.com/pyg-team/pytorch_geometric)
- [Colab Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)
