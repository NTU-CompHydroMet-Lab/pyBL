# Building Documentation

Install doc dependencies and serve locally:

```bash
uv sync --group doc
uv run mkdocs serve
```

Build static output:

```bash
uv run mkdocs build
```

Output is written to `site/`.
