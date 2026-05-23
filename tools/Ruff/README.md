# Ruff — Python Linter and Code Formatter

**Version:** 0.6.x (stable)

## Purpose

Ruff is an extremely fast Python linter and code formatter written in Rust. It is designed as a drop-in replacement for Flake8 (plus dozens of plugins), isort, pydocstyle, pyupgrade, and yes, even Black. Key advantages:

- **Speed** — Ruff is 10–100× faster than traditional linting tools. It can lint an entire codebase in milliseconds.
- **Broad rule coverage** — over 800 lint rules covering style, correctness, complexity, security, and more.
- **Autofix** — many violations can be fixed automatically (`ruff check --fix`).
- **Formatter** — `ruff format` is (mostly) compatible with Black but faster.
- **Single dependency** — replaces flake8 + isort + pydocstyle + pyupgrade + autoflake + bandit, and more.
- **pyproject.toml native** — all configuration lives in one file.
- **Pre-commit support** — native hooks available.

In the WFB model project, Ruff provides both linting and formatting, catching bugs, enforcing style, and standardising imports — all in one tool.

## Installation

```bash
pip install ruff==0.6.9

# Conda:
conda install -c conda-forge ruff=0.6.9
```

**Platform notes:**
- Pre-built wheels available for Windows, macOS, and Linux (x86_64 and ARM64).
- No Rust toolchain required — Ruff is distributed as a binary wheel.

## Basic Usage

```bash
# Lint the project
ruff check src/ tests/

# Lint with automatic fix
ruff check --fix src/

# Show which rules would be triggered without fixing
ruff check --diff src/

# Format code (Black-compatible)
ruff format src/ tests/

# Check formatting without writing changes
ruff format --check src/

# Sort and organise imports (included in `ruff check --fix`)
ruff check --select I --fix src/
```

**Output example:**

```
src/train.py:42:5: T201 `print` found
src/train.py:88:13: F841 local variable `unused` is assigned to but never used
Found 2 errors.
```

**One-command lint + format:**

```bash
ruff check src/ --fix && ruff format src/
```

## Advanced Usage / Configuration

| Command / Option | Description |
|---|---|
| `ruff check` | Lint the given files / directories. |
| `ruff check --fix` | Apply autofixes for fixable violations. |
| `ruff check --select <RULE>` | Enable only specific rules or categories. |
| `ruff check --ignore <RULE>` | Disable specific rules. |
| `ruff check --add-noqa` | Add `# noqa` annotations for all current violations. |
| `ruff format` | Format code to Ruff's style (Black-compatible). |
| `ruff format --check` | Check formatting without modifying files. |
| `ruff format --line-length <N>` | Override line length (default 88). |
| `ruff check --watch` | Watch mode — re-lint on file changes. |

**Configuration in `pyproject.toml`:**

```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "SIM", "PL", "RUF"]
ignore = ["E501"]  # line-too-long (handled by formatter)

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]  # allow `assert` in tests
"src/__init__.py" = ["F401"]  # unused import OK in __init__

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false

[tool.ruff.lint.isort]
known-first-party = ["src"]
```

| Rule Category | Prefix | Description |
|---|---|---|
| Pycodestyle | `E`, `W` | PEP 8 style violations. |
| Pyflakes | `F` | Logical errors (unused imports, undefined names). |
| isort | `I` | Import order. |
| pyupgrade | `UP` | Modern Python idioms. |
| flake8-simplify | `SIM` | Simpler, more readable code. |
| Pylint | `PL` | Pylint rules ported to Ruff. |
| pep8-naming | `N` | Naming conventions. |
| flake8-bugbear | `B` | Likely bugs. |
| Ruff-specific | `RUF` | Ruff-specific rules. |

## Integration with the WFB Model Project

Ruff is the primary code quality tool in the WFB model project:

1. **Linting** — `ruff check` runs in CI on every PR (via GitHub Actions) and catches issues like unused imports, undefined names, and non-Pythonic patterns.
2. **Formatting** — `ruff format` replaces Black as the formatter (faster, same style). Configuration is shared in `pyproject.toml`.
3. **Import sorting** — the `I` rule automatically organises imports into standard library, third-party, and first-party sections.
4. **Pre-commit hook** — Ruff runs in `.pre-commit-config.yaml`, ensuring every commit is linted and formatted.
5. **No config duplication** — all settings live in `pyproject.toml`, replacing `setup.cfg`, `.flake8`, `.isort.cfg`, etc.

## Common Pitfalls / Troubleshooting

- **Ruff vs flake8 plugin X** — if you previously used a flake8 plugin not yet implemented in Ruff, check the [rule parity table](https://docs.astral.sh/ruff/faq/#does-ruff-support-x) in the FAQ.
- **`ruff format` differs from Black** — Ruff's formatter aims for ~99% compatibility. Known differences include magic trailing comma handling and some nested parentheses. Run `ruff format` instead of `black` consistently.
- **Missing rule codes** — run `ruff rule <CODE>` to see the full explanation and examples for any rule.
- **Performance** — Ruff is extremely fast, but the first run on a cold cache may be slightly slower. Use `--show-fixes` to see applied fixes.
- **False positives** — suppress individual lines with `# noqa: <RULE>` or disable rules per file in config.

## Documentation Links

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ruff Rules](https://docs.astral.sh/ruff/rules/)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)
- [Ruff Linter](https://docs.astral.sh/ruff/linter/)
- [Ruff Formatter](https://docs.astral.sh/ruff/formatter/)
- [Ruff pre-commit hook](https://github.com/astral-sh/ruff-pre-commit)
