# Dreamcatcher — Living Memory for AI Agents

## Session Saving (Required Behavior)

At the end of every meaningful conversation — one that contains personal information, project updates, decisions, preferences, or any facts worth remembering — you MUST call `dreamcatcher_save_session` with the full conversation transcript (both user messages and your responses).

Skip saving only for trivial single-command interactions with no personal content.

This is not optional. Session transcripts are the raw material for nightly memory consolidation. If you don't save them, the memory model can't learn from them.

## Available Tools

- **`dreamcatcher_recall`** — Query specific memories about the user (projects, preferences, relationships, decisions)
- **`dreamcatcher_status`** — Check memory model health, memory counts, training history
- **`dreamcatcher_save_session`** — Save conversation transcript for nightly training

## Project Structure

```
dreamcatcher/
  __init__.py          # Package init, version
  __main__.py          # CLI entry point (all commands)
  config.py            # DreamcatcherConfig dataclass, YAML loading
  database.py          # MemoryDB — SQLite storage for sessions, memories, training examples
  collector.py         # SessionCollector (ingest + LLM extraction) + TrainingDataBuilder
  trainer.py           # MemoryTrainer — dual backend (MLX on Apple Silicon, PyTorch on NVIDIA)
  server.py            # FastAPI inference server (:8420)
  mcp_server.py        # MCP bridge for Claude Code (stdio transport)
  wiki.py              # WikiExporter — Obsidian-compatible markdown vault
  lint.py              # MemoryLinter — rule-based + LLM consistency checks
dreamcatcher_client.py # Single-file HTTP client (copy into any agent project)
config.yaml            # Default configuration
```

## Architecture

- **Daytime**: Sessions are saved to SQLite via HTTP POST. No learning happens.
- **3 AM Nightly Pipeline**: Extract memories (frontier LLM) -> Build training set (with semantic compression) -> Re-fine-tune from base weights -> Deploy via atomic symlink swap
- **Morning**: Agents query the local memory model. It returns structured JSON for prompt injection.

## Key Invariants

- `M_t = F(M_0, T_t)` — model is always fine-tuned from fixed base weights, never from a checkpoint
- Canonical ledger (`C_t`) is append-only — no extracted fact is ever removed
- Semantic compression is a rendering policy, not a deletion operation
- The memory model runs on-device; only structured context leaves the device

## Development

```bash
pip install -e .                    # Server + client
pip install mlx mlx-lm anthropic   # Apple Silicon training
pip install -e ".[train]"          # NVIDIA training
pip install -e ".[claude-code]"    # MCP integration
```

## Testing Changes

```bash
pytest tests/ -v                   # Run unit tests
dreamcatcher stats                 # Check current state
dreamcatcher nightly               # Run full pipeline manually
dreamcatcher lint --rules-only     # Zero-cost consistency check
dreamcatcher wiki                  # Export vault for inspection
```

## Client Library

The HTTP client class is `LivingMemory` (in `dreamcatcher_client.py`). Backward-compat aliases `PersonalMemory` and `DreamcatcherMemory` are available.
