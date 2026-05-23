# Tokenizer Training

**Purpose:** Train a custom tokenizer on your dataset for optimal vocabulary coverage.

## Why Train Your Own?

- **Better coverage** — domain-specific vocabulary (code, medical, legal)
- **Smaller vocabulary** — fewer tokens for common terms in your domain
- **No special tokens mismatch** — full control over token IDs

## Installation

```powershell
pip install tokenizers sentencepiece
```

## Training Script

```python
# train_tokenizer.py
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from datasets import load_dataset

# 1. Create BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 2. Normalization
tokenizer.normalizer = normalizers.NFKC()

# 3. Pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 4. Post-processor
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
tokenizer.decoder = decoders.ByteLevel()

# 5. Trainer
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    min_frequency=2,
    show_progress=True,
)

# 6. Load data
dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

def batch_iterator(batch_size=1000):
    iterator = iter(dataset)
    while True:
        try:
            yield [next(iterator)["text"] for _ in range(batch_size)]
        except StopIteration:
            break

# 7. Train
print("Training tokenizer...")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# 8. Save
tokenizer.save("tokenizer.json")
print("Tokenizer saved to tokenizer.json")

# 9. Convert for HuggingFace
from transformers import PreTrainedTokenizerFast
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
)
hf_tokenizer.save_pretrained("hf-tokenizer")
print("HF tokenizer saved to hf-tokenizer/")
```

## Choosing Vocab Size

| Domain | Recommended Vocab Size |
|--------|----------------------|
| General English | 32,000 - 50,000 |
| Code | 32,000 - 64,000 |
| Multilingual | 64,000 - 128,000 |
| Chinese/Japanese/Korean | 64,000 - 128,000 |
| Medical/Scientific | 50,000 - 100,000 |

## Testing Your Tokenizer

```python
tokenizer = PreTrainedTokenizerFast.from_pretrained("hf-tokenizer")
test = "Hello, world! This is a test."
tokens = tokenizer.tokenize(test)
ids = tokenizer.encode(test)
print(f"Tokens: {tokens}")
print(f"IDs: {ids}")
print(f"Compression ratio: {len(test) / len(ids):.2f} chars/token")
```
