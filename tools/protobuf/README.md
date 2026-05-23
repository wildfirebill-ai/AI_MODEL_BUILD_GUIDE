# Protocol Buffers (protobuf)

**Version:** 4.25+ (check compatibility — v21+ renamed to `protobuf`; older code uses `google.protobuf`)

**Purpose:** Language-neutral, platform-neutral extensible mechanism for serializing structured data. Protobuf is used extensively in ML infrastructure: model serialization (TensorFlow SavedModel, ONNX), tokenizer configurations (SentencePiece), and distributed training deployment specs (DeepSpeed, Megatron).

## Installation

```powershell
pip install protobuf
```

Pin to a version compatible with your framework:
```powershell
pip install "protobuf>=4.25,<5"
```

### Notes
- TensorFlow requires a specific protobuf version range — mismatches cause cryptic import errors
- Do **not** install `google-protobuf` (an old, deprecated package) — use `protobuf` only
- The `protobuf` package supplies the Python runtime; the `protoc` compiler must be installed separately (https://github.com/protocolbuffers/protobuf/releases)

## Basic Usage

```python
import protobuf  # noqa: F401 — import ensures `.proto` descriptors are registered
from google.protobuf import json_format

# Convert between protobuf and JSON (useful for inspection)
# protobuf is typically consumed indirectly through libraries
import sentencepiece as spm

# Train a SentencePiece model (uses protobuf internally)
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="m",
    vocab_size=32000,
)
# Produces m.model (protobuf format)
```

## Advanced Usage / Configuration
- **Schema evolution:** Protobuf supports field addition/removal without breaking backward compatibility — crucial for long-lived model formats
- **Descriptor pools:** If loading multiple `.proto` files at runtime, use `google.protobuf.descriptor_pool.Default()` to avoid duplicates
- **Custom ops:** DeepSpeed and Megatron use protobuf for deployment configuration — `.prototxt` files define model parallelism layouts
- **ONNX models** are protobuf serialized — `onnx.load("model.onnx")` deserializes via protobuf

## Integration with WFB Model
Protobuf is a transitive dependency pulled in by ONNX, TensorFlow (if used), SentencePiece tokenizer training, and DeepSpeed configuration parsing. The WFB model's tokenizer training script (`scripts/train_tokenizer.py`) produces protobuf-format model files via SentencePiece.

## Common Pitfalls / Troubleshooting
- **`ImportError: cannot import name 'builder' from 'google.protobuf'`:** Version mismatch — upgrade protobuf: `pip install --upgrade protobuf`
- **`TypeError: Couldn't build proto file into descriptor pool`:** The `.proto` file path or syntax is wrong — check file paths and proto syntax version
- **Protobuf uses more memory than JSON:** This is expected — protobuf keeps descriptors in memory for fast access; use `SerializeToString()` and `ParseFromString()` for streaming
- **Protobuf 4.x breaking changes:** Code written for protobuf 3.x may need updates — check `google.protobuf.text_format` and descriptor API changes
- **Conflict with `pydantic` validation:** Some libraries mix protobuf and pydantic — ensure serialization round-trips correctly by comparing JSON outputs

## Platform-specific Notes
- **Windows:** Pre-built wheels available for all protobuf versions — no compiler needed
- **Linux/macOS:** Some protobuf versions compile C++ extensions on install — ensure `gcc`/`clang` is available
- **Alpine Linux:** May require `apk add protobuf-dev` for the C++ proto compiler

## Integration with WFB Model
Protobuf is a transitive dependency in the WFB model project:
- **ONNX export/import:** `scripts/export_onnx.py` uses protobuf for model serialization
- **SentencePiece tokenizer:** Training produces `.model` protobuf files consumed by `scripts/train_tokenizer.py`
- **DeepSpeed configs:** Deployment configuration `.prototxt` files are parsed via protobuf

## Troubleshooting Version Conflicts
```powershell
# Check installed protobuf version
pip show protobuf

# If TensorFlow pins a specific range, match it exactly
pip install "protobuf>=3.20,<4"

# For ONNX, protobuf 4.x is required
pip install "protobuf>=4.21,<5"
```

## Documentation
- https://protobuf.dev/
- https://github.com/protocolbuffers/protobuf/tree/main/python
