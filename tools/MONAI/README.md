# MONAI — Medical Open Network for AI

PyTorch-based framework for medical image analysis: segmentation, classification, and preprocessing.

## Key Components

| Component | Description |
|-----------|-------------|
| `monai.networks.nets.UNet` | 2D/3D U-Net for segmentation |
| `monai.networks.nets.DenseNet121` | 3D DenseNet for classification |
| `monai.transforms` | Medical image preprocessing |
| `monai.losses.DiceLoss` | Dice and combined Dice-CE losses |
| `monai.metrics.DiceMetric` | Dice scoring |
| `monai.inferers.SlidingWindowInferer` | Patch-based large volume inference |

## Installation

```bash
pip install monai
pip install monai[cuda121]  # CUDA support
```

## 3D U-Net Segmentation

```python
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import Compose, ScaleIntensityd, RandRotate90d, RandFlipd, ToTensord
from monai.data import DataLoader, Dataset

images = torch.randn(10, 1, 64, 64, 64)
labels = (torch.rand(10, 1, 64, 64, 64) > 0.8).float()
data = [{"img": img, "seg": lbl} for img, lbl in zip(images, labels)]

transforms = Compose([ScaleIntensityd(keys=["img"]),
                      RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 1)),
                      RandFlipd(keys=["img", "seg"], prob=0.5), ToTensord(keys=["img", "seg"])])
loader = DataLoader(Dataset(data=data, transform=transforms), batch_size=2, shuffle=True)

model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
loss_fn = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    loss_sum = 0
    for batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(batch["img"]), batch["seg"])
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1} | Loss: {loss_sum/len(loader):.4f}")
```

## DenseNet, Sliding Window & Transforms

```python
from monai.networks.nets import DenseNet121
from monai.inferers import SlidingWindowInferer
from monai.transforms import (LoadImaged, Orientationd, Spacingd,
                               NormalizeIntensityd, RandCropByPosNegLabeld)

model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)

inferer = SlidingWindowInferer(roi_size=(32, 32, 32), sw_batch_size=4, overlap=0.5)
with torch.no_grad():
    output = inferer(input_volume, model)

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                           spatial_size=(96, 96, 96), pos=2, neg=1, num_samples=4),
])
```

## Dice Metric & Lightning

```python
from monai.metrics import DiceMetric

metric = DiceMetric(include_background=False, reduction="mean_batch")
for batch in loader:
    with torch.no_grad():
        metric(y_pred=(model(batch["img"]) > 0.5).float(), y=batch["seg"])
print(f"Dice: {metric.aggregate()}")

import pytorch_lightning as pl
class MONAISegmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(3, 1, 1, (16, 32, 64, 128, 256), (2, 2, 2, 2))
        self.loss = DiceLoss(sigmoid=True)
    def training_step(self, batch, batch_idx):
        loss = self.loss(self.model(batch["img"]), batch["seg"])
        self.log("train_loss", loss); return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

pl.Trainer(max_epochs=50, accelerator="gpu", devices=1).fit(MONAISegmenter(), loader)
```

## Resources

- [MONAI Docs](https://docs.monai.io/)
- [GitHub](https://github.com/Project-MONAI/MONAI)
- [Tutorials](https://github.com/Project-MONAI/tutorials)
- [Model Zoo](https://monai.io/model-zoo.html)
