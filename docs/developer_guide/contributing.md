# Contributing to pyBL

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

**Install uv**

Follow the installation instructions at [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).
uv manages both Python versions and virtual environments — no separate pyenv or virtualenv needed.

**Clone the repository**

```bash
git clone https://github.com/NTU-CompHydroMet-Lab/pyBL.git
cd pyBL
```

**Install dependencies**

```bash
uv sync --group dev
```

This creates a virtual environment and installs all development dependencies automatically.

## Verify your setup

Run the test suite:

```bash
uv run pytest
```

Run the linters:

```bash
uv run ruff check pybl
uv run mypy pybl
```

If the tests and linters pass, you are ready to contribute.
