# Tabularasa

**Purpose:** Synthetic instruction data generation and quality filtering for LLM fine-tuning.

## Installation

```powershell
pip install tabularasa
```

## Usage

```python
from tabularasa import DataGenerator, QualityFilter, DataAugmenter

# 1. Generate synthetic instructions
generator = DataGenerator(
    model="gpt-4",  # Teacher model for generation
    api_key="...",
)

instructions = generator.generate(
    topics=["python", "math", "writing"],
    n_per_topic=100,
    output_format="sharegpt",  # ShareGPT format
)

# 2. Filter for quality
filter = QualityFilter(
    min_length=50,
    max_length=2048,
    remove_duplicates=True,
    filter_toxicity=True,
)
cleaned = filter.filter(instructions)

# 3. Augment with variations
augmenter = DataAugmenter()
augmented = augmenter.augment(
    cleaned,
    techniques=["paraphrase", "back_translate"],
)
```

## Key Features

- **Topic-based generation** of instruction data
- **Quality filters** (length, toxicity, repetition, uniqueness)
- **Data augmentation** (paraphrasing, back-translation)
- **Format conversion** (ShareGPT, Alpaca, ChatML, etc.)
- **System prompt generation** for role-playing data

## Documentation

- https://github.com/glaiveai/tabularasa
