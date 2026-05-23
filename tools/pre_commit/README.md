# pre-commit — Git Hook Framework for Automated Checks

**Version:** 3.8.x (stable)

## Purpose

pre-commit is a multi-language package manager for Git hook scripts. It allows you to define a set of automated checks (linters, formatters, security scanners, etc.) that run before every `git commit` — or on other Git events (pre-push, post-checkout, etc.). If any check fails, the commit is blocked until the issue is fixed. Key features:

- **Language-agnostic** — hooks can be Python, Node, Ruby, Docker, Rust, or system binaries.
- **Managed environments** — each hook runs in its own isolated environment (virtualenv, npx, Docker).
- **Caching** — hooks are cached and reused across commits for performance.
- **Incremental adoption** — `--from-ref` / `--to-ref` flags allow checking only changed files.
- **Repository-level configuration** — a single `.pre-commit-config.yaml` file declares all hooks with version pins.

In the WFB model project, pre-commit is the gatekeeper that enforces code quality standards (Black/Ruff formatting, linting, type checks) before any code enters the repository.

## Installation

```bash
pip install pre-commit==3.8.0

# Conda:
conda install -c conda-forge pre-commit=3.8.0

# Or via package manager:
# macOS: brew install pre-commit
# Linux: apt install pre-commit (may be outdated)
```

**Platform notes:**
- All platforms supported. Windows may need `git config core.autocrlf true` to handle line endings correctly.
- After install, run `pre-commit install` inside the repository to activate hooks.

## Basic Usage

```bash
# Install hooks into .git/hooks/
pre-commit install

# Run all hooks on all files (initial setup)
pre-commit run --all-files

# Run hooks only on staged files (same as commit-time behaviour)
pre-commit run

# Run a specific hook
pre-commit run ruff --all-files

# Skip hooks on a commit (use sparingly)
git commit -m "wip" --no-verify

# Update all hooks to their latest tagged versions
pre-commit autoupdate
```

**Sample `.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [numpy, pandas, torch]
```

## Advanced Usage / Configuration

| Option / Command | Description |
|---|---|
| `rev:` | Pin a specific hook version (tag, commit SHA, or branch). **Always pin.** |
| `args:` | Command-line arguments passed to the hook. |
| `additional_dependencies:` | Extra packages for the hook's environment. |
| `files:` / `exclude:` | Regex patterns to limit which files the hook runs on. |
| `stages:` | Run hook only on specific stages: `commit`, `push`, `merge-commit`, `manual`. |
| `--all-files` | Run hooks on every file in the repo (not just staged). |
| `--from-ref` / `--to-ref` | Run hooks on files changed between two Git refs (CI use). |
| `pre-commit autoupdate` | Update hooks to the latest tags and create a diff for review. |
| `pre-commit clean` | Remove cached hook environments. |

**Running in CI (GitHub Actions):**

```yaml
- name: Run pre-commit
  uses: pre-commit/action@v3.0.1
```

**Example with `stages:` — only run on push, not commit:**

```yaml
- repo: https://github.com/jumanjihouse/pre-commit-hooks
  rev: 3.0.0
  hooks:
    - id: shellcheck
      stages: [push]
```

## Integration with the WFB Model Project

pre-commit is the first line of defense for code quality in the WFB model project:

1. **Pre-commit stage** — every developer runs `pre-commit install` after cloning. Before every commit, Ruff lints and formats, mypy checks types, and basic sanity checks (trailing whitespace, YAML validity, large files) are run.
2. **Consistent environment** — hooks are version-pinned in `.pre-commit-config.yaml`, ensuring all team members and CI use the same tool versions.
3. **CI integration** — the `pre-commit/action` runs in GitHub Actions on every PR, catching issues that were committed with `--no-verify`.
4. **Manual stage** — some hooks (e.g., expensive integration tests) are configured with `stages: [manual]` and run on demand via `pre-commit run --hook-stage manual`.
5. **No config drift** — `pre-commit autoupdate` is run periodically and committed, keeping hooks current.

## Common Pitfalls / Troubleshooting

- **Hooks fail on Windows** — line ending issues are common. Set `git config core.autocrlf true` and ensure `.pre-commit-config.yaml` uses `lf` line endings. Add `- id: mixed-line-ending` with `args: [--fix=lf]`.
- **Hook slow for large repos** — use `stages: [push]` for expensive hooks, or configure `files:` regex to limit scope.
- **`pre-commit` not found after install** — ensure the Python scripts directory is on your PATH (`pip install` adds it).
- **Hooks run on unchanged files** — by default, pre-commit only runs on staged files. If a hook runs on unchanged files, check that `pass_filenames: true` (default) is set and the tool accepts file arguments.
- **Version mismatch across team** — always pin `rev:` to a specific tag or SHA. Never use `rev: main` or `rev: v3.x`.
- **`pre-commit autoupdate` breaks something** — review the diff created by `autoupdate` before committing. A new major version of a hook may have breaking changes.

## Documentation Links

- [pre-commit Documentation](https://pre-commit.com/)
- [pre-commit Configuration](https://pre-commit.com/#config-pre-commit-config-yaml)
- [pre-commit Hooks — Official Collection](https://github.com/pre-commit/pre-commit-hooks)
- [pre-commit in CI](https://pre-commit.com/#usage-in-continuous-integration)
- [Available Hooks](https://pre-commit.com/hooks.html)
