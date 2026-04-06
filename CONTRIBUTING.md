# Contributing to Dreamcatcher

Thanks for your interest in contributing to Dreamcatcher. This document covers the basics.

## Development Setup

```bash
git clone https://github.com/alexskatell/living-memory.git
cd living-memory

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all deps
pip install -e ".[all]"
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

> **Note (uv / Python 3.12):** If you installed Python via [uv](https://github.com/astral-sh/uv), editable installs (`-e`) may silently break — Python's `site` module skips `.pth` files starting with `_`, preventing the package from being found outside the project directory. This particularly affects the MCP server, which runs from an arbitrary working directory. If you hit `ModuleNotFoundError: No module named 'dreamcatcher'`, reinstall without `-e`: `pip install ".[all]"`

## Project Structure

```
dreamcatcher/          # Core package
  config.py            # Configuration dataclasses
  database.py          # SQLite storage layer
  collector.py         # Session ingestion + LLM extraction
  trainer.py           # Dual-backend training (MLX / PyTorch)
  server.py            # FastAPI inference server
  mcp_server.py        # Claude Code MCP bridge
  wiki.py              # Obsidian vault export
  lint.py              # Memory consistency checker
dreamcatcher_client.py # Single-file HTTP client
tests/                 # Test suite
```

## Key Invariants

These must be preserved in all changes:

1. **No sequential-update drift**: `M_t = F(M_0, T_t)` — the model is always trained from fixed base weights
2. **Append-only canonical ledger**: No extracted fact is ever deleted from SQLite
3. **Semantic compression is a rendering policy**: Compression affects what's included in the training set, never what's stored
4. **On-device inference**: The memory model runs locally; only structured context leaves the device
5. **Linter never mutates**: The linting system produces advisory reports, never silently changes data

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just unit tests (no API calls)
pytest tests/ -v -k "not integration"
```

## Code Style

- Follow existing patterns in the codebase
- Type hints on public API methods
- Docstrings on classes and non-trivial functions
- Keep the core package dependency-light (no GPU deps in the server path)

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Run `pytest tests/ -v` and ensure all tests pass
5. Open a PR with a clear description of what changed and why

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (OS, Python version, hardware)
