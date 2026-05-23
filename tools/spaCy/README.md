# spaCy — Industrial-Strength NLP

Advanced NLP with pretrained pipelines for tokenization, tagging, parsing, NER, and lemmatization.

## Installation

```bash
pip install spacy
python -m spacy download en_core_web_lg
```

## Quick Start

```python
import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")

for token in doc:
    print(f"{token.text:<12} {token.lemma_:<10} {token.pos_:<8} {token.dep_:<10}")

for ent in doc.ents:
    print(f"{ent.text:<30} {ent.label_:<10} {ent.start_char}-{ent.end_char}")
```

## Pipeline Components

```python
print(nlp.pipe_names)  # ['tok2vec', 'tagger', 'parser', 'ner', 'lemmatizer']
```

### Custom Trainable Component

```python
@spacy.registry.architectures("my_cat_tagger.v1")
def build_cat_tagger():
    from spacy.ml import build_bow_text_classifier
    return build_bow_text_classifier(nO=2, ngram_size=1)

nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

train_data = [
    ("This product is amazing", {"cats": {"POSITIVE": 1, "NEGATIVE": 0}}),
    ("Terrible experience", {"cats": {"POSITIVE": 0, "NEGATIVE": 1}}),
]

optimizer = nlp.begin_training()
for epoch in range(10):
    for text, cats in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"cats": cats})
        nlp.update([example], sgd=optimizer)

doc = nlp("Really great service")
print(doc.cats)
```

### Text Preprocessing Pipeline

```python
def preprocess(texts, nlp, batch_size=256):
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct]
        entities = [(e.text, e.label_) for e in doc.ents]
        yield {"tokens": tokens, "entities": entities}

for result in preprocess(["Apple released the new MacBook Pro."], nlp):
    print(result)
```

### Rule-Based Matching

```python
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
matcher.add("ELON_MUSK", [[{"LOWER": "elon"}, {"LOWER": "musk"}]])
doc = nlp("Elon Musk founded SpaceX.")
print([doc[start:end].text for _, start, end in matcher(doc)])
```

## Key Classes

| Class | Description |
|-------|-------------|
| `Doc` | Sequence of tokens |
| `Token` | Word with linguistic features |
| `Span` | Slice of a Doc |
| `Example` | Training data wrapper |

## Integration

Use `nlp.pipe()` for batched processing. Add custom components with `nlp.add_pipe()`. Extract features for downstream models.
