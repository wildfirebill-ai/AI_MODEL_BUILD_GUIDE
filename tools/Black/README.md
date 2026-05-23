# Black — The Uncompromising Python Code Formatter

**Version:** 24.8.x (stable)

## Purpose

Black is an opinionated Python code formatter that automatically reformats code to conform to a consistent style. By design, it offers minimal configuration — the goal is to eliminate debates about formatting and produce readable, diff-friendly output. Key characteristics:

- **Deterministic** — given the same input and line length, Black always produces the same output.
- **Minimal configuration** — line length (default 88) and string normalisation (`"` vs `'`) are the main knobs.
- **Fast** — formats large codebases in seconds.
- **PEP 8 compliant** — Black's style is a strict subset of PEP 8 with a few deliberate deviations (longer line length, more aggressive line breaks).
- **IDE integration** — works with VSCode, PyCharm, Vim, Emacs, and pre-commit hooks.
- **Stable mode** — `--stable` guarantees output does not change across minor versions for the same code.

In the WFB model project, Black ensures all Python source files adhere to a uniform style, improving code review efficiency and reducing formatting noise in diffs.

## Installation

```bash
pip install black==24.8.0

# Conda:
conda install -c conda-forge black=24.8.0

# Optional: Jupyter notebook support
pip install "black[jupyter]==24.8.0"
```

**Platform notes:**
- All platforms behave identically.
- On Windows, line endings are normalised to the platform default; use `--line-length` consistently across environments.

## Basic Usage

```bash
# Format a single file
black src/train.py

# Format an entire directory (recursively)
black src/ tests/

# Check formatting without writing changes
black --check src/

# See diff of what would change
black --diff src/

# Format with a custom line length
black --line-length 100 src/
```

**Before Black:**

```python
def predict(model,    input_ids, attention_mask):
    logits=model(input_ids, attention_mask)
    probs = logits.softmax(dim=-1)
    return probs.argmax(dim=-1),probs.max(dim=-1).values
```

**After Black:**

```python
def predict(model, input_ids, attention_mask):
    logits = model(input_ids, attention_mask)
    probs = logits.softmax(dim=-1)
    return probs.argmax(dim=-1), probs.max(dim=-1).values
```

## Advanced Usage / Configuration

| Option | Description |
|---|---|
| `--line-length` | Target line length (default 88). Shorter = more wrapping. |
| `--target-version` | Python version target (`py39`, `py310`, `py311`). Affects syntax. |
| `--skip-string-normalization` | Do not normalise string quotes (preserve existing `"` / `'`). |
| `--preview` | Enable preview style (upcoming formatting changes). |
| `--fast` / `--safe` | Safe mode (default) checks AST after formatting. `--fast` skips this. |
| `--quiet` / `--verbose` | Control output verbosity. |
| `--include` / `--exclude` | Regex patterns for file inclusion / exclusion. |

**Configuration file `pyproject.toml`:**

```toml
[tool.black]
line-length = 88
target-version = ["py311"]
skip-string-normalization = false
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''
```

**CI integration:**

```yaml
# .github/workflows/ci.yml (excerpt)
- name: Check formatting
  run: black --check --diff src/ tests/
```

## Integration with the WFB Model Project

Black enforces code style consistency across the WFB model codebase:

1. **Pre-commit hook** — Black runs via `pre-commit` before every commit, preventing unformatted code from entering the repository.
2. **CI gate** — the CI pipeline runs `black --check` on every PR. PRs with unformatted code are blocked.
3. **Editor integration** — team members use Black as their editor formatter (format-on-save), so formatting is automatic during development.
4. **Consistent diffs** — because Black normalises formatting, code reviews focus on logic changes, not whitespace.

## Common Pitfalls / Troubleshooting

- **`black --check` fails for third-party code** — use Black's `--exclude` to skip vendored or generated directories. Add `exclude = '''(\.venv\|build\|dist)'''` to `pyproject.toml`.
- **Trailing commas** — by design, Black adds a trailing comma when a collection is broken across multiple lines. This is a feature, not a bug (it creates cleaner diffs when adding new items).
- **Magic trailing comma** — if you add a trailing comma yourself, Black will force the collection to break across multiple lines. Omit the trailing comma to let Black decide.
- **No way to disable for a block** — Black intentionally does not support `# fmt: off` / `# fmt: on` unless you use them sparingly. For very specific cases you can use `# fmt: off` / `# fmt: on` markers.
- **Version mismatch** — ensure all developers and CI use the same Black version to avoid format conflicts.

## Documentation Links

- [Black Documentation](https://black.readthedocs.io/en/stable/)
- [Black GitHub](https://github.com/psf/black)
- [Black Configuration](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html)
- [Black in pre-commit](https://black.readthedocs.io/en/stable/guides/using_black_with_pre_commit.html)
- [Black Editor Integration](https://black.readthedocs.io/en/stable/integrations/editors.html)
