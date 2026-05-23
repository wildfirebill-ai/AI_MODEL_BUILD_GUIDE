# SentencePiece & HuggingFace Tokenizers

**Purpose:** Tokenization — converting text to token IDs and vice versa.

---

## SentencePiece

### Installation

```powershell
pip install sentencepiece
```

### Usage

```python
import sentencepiece as spm

# Train a tokenizer
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=32000,
    model_type="bpe",  # or "unigram"
    character_coverage=1.0,
    max_sentence_length=2048,
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")
ids = sp.encode("Hello, world!")
text = sp.decode(ids)
```

---

## HuggingFace Tokenizers (higher level)

### Installation

```powershell
pip install tokenizers
```

### Training a BPE Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from datasets import load_dataset

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    min_frequency=2,
)

dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

def batch_iterator(batch_size=1000):
    for i in range(0, 100000, batch_size):
        yield [next(iter(dataset))["text"] for _ in range(batch_size)]

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.save("tokenizer.json")

# Load
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer.save_pretrained("my-tokenizer")
```

### Key Configurations

| Vocab Size | Use Case |
|-----------|----------|
| 8,000 | Small models, speed critical |
| 32,000 | Standard — Mistral, LLaMA 2 |
| 50,000 | Larger — GPT-4 |
| 128,000 | Very large — LLaMA 3 |
