# fastText

**Purpose:** Efficient text classification and word representation library -- used for quality filtering of training data.

## Installation

```powershell
pip install fasttext
```

## Usage for Data Quality Classification

```python
import fasttext

# Train a quality classifier
# Data format: "__label__high <text>" / "__label__low <text>"
model = fasttext.train_supervised(
    input="quality_data.txt",
    lr=0.1,
    epoch=25,
    wordNgrams=2,   # Use bigrams
    dim=256,        # Embedding dimension
    loss="softmax", # or "hs" for hierarchical softmax
)

# Score documents
def score_document(text):
    labels, scores = model.predict(text.replace("\n", " "))
    return scores[0] if labels[0] == "__label__high" else 1 - scores[0]

# Filter: keep top 70%
dataset_scores = [(i, score_document(d["text"])) for i, d in enumerate(dataset)]
threshold = sorted(dataset_scores, key=lambda x: x[1])[int(len(dataset_scores) * 0.3)][1]
filtered = [dataset[i] for i, s in dataset_scores if s >= threshold]

# Save and load
model.save_model("quality_filter.bin")
model = fasttext.load_model("quality_filter.bin")
```

## Key Features

- **Extremely fast** training (minutes on millions of docs)
- **Sublinear scaling** with data size
- **Word n-gram features** capture partial word information
- **Supervised and unsupervised** modes

## Documentation

- https://fasttext.cc/
