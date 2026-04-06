# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-06

### Added
- **Claude Code MCP integration** — Native Model Context Protocol server (`dreamcatcher mcp`) with stdio transport, one-command setup (`dreamcatcher setup claude-code --global`), and personal context injection at session start
- **MCP tools** — `living_memory_recall`, `living_memory_status`, `living_memory_save_session` for on-demand memory queries, health checks, and automatic session capture
- **Test suite** — 64 pytest tests covering database, config, collector, server, and client (all passing)
- **GitHub Actions CI** — Automated testing on Python 3.10/3.11/3.12 matrix on every push/PR
- **CLAUDE.md** — Mandatory session-saving instructions for Claude Code users working in this repo
- **CHANGELOG.md** — Version history tracking
- **CONTRIBUTING.md** — Contribution guidelines and development setup
- **`ExtractionConfig` dataclass** — Configurable extraction provider and model (previously hardcoded)
- **`model_age_hours`** in `/health` endpoint response for monitoring model freshness
- **`config.ensure_dirs()`** method — Explicit directory creation (replaces implicit side effect on config load)
- **`WikiExporter` and `MemoryLinter`** added to `__all__` exports in `__init__.py`
- **`LivingMemory` client class** — Primary HTTP client (renamed from `PersonalMemory`), with backward-compatibility aliases `PersonalMemory` and `DreamcatcherMemory`
- **Multi-stage Dockerfile** — Separate `server` (lightweight, no GPU deps) and `training` (PyTorch + cron) build targets
- **`integrations/claude-code/README.md`** — Detailed integration guide with architecture diagram, tool reference, and troubleshooting

### Changed
- **MCP server renamed** — Server name `"dreamcatcher"` -> `"living-memory"`, tool prefix `dreamcatcher_` -> `living_memory_`
- **Client class renamed** — `PersonalMemory` -> `LivingMemory` (old names still work as aliases)
- **Extraction model reads from config** — `collector.py` now uses `config.extraction.model` instead of hardcoding `claude-sonnet-4-20250514`
- **`DreamcatcherConfig.load()`** no longer creates directories as a side effect
- **Docker Compose** updated to use multi-stage build targets
- **README quickstart** updated with Claude Code MCP setup instructions

### Fixed
- `pyproject.toml` placeholder URLs (`[username]`) replaced with `alexskatell/living-memory`
- `pyproject.toml` author name placeholder replaced with `Alex Skatell`
- `docs/x-post.md` `[link]` placeholders replaced with actual GitHub URLs
- `build/` and `egg-info/` directories removed from git tracking
- `.gitignore` now covers `.venv/` and `dreamcatcher_memory.egg-info/`
- Server keyword search has TODO for upgrade to embedding-based search

## [0.1.0] - 2026-04-04

### Added
- Initial release of the Dreamcatcher architecture
- Core server with collector, trainer, wiki builder, and linter
- Nightly pipeline: extract via frontier LLM, re-fine-tune from scratch, benchmark, atomic deploy
- Hermes Agent integration (Python lifecycle hooks)
- OpenClaw integration (TypeScript memory slot plugin)
- Client library (`dreamcatcher_client.py`) for any Python agent
- Docker support with cron-based nightly training
- Dual-backend trainer (MLX on Apple Silicon, PyTorch on NVIDIA)
- Semantic compression for training set rendering
- Disaster recovery delta context injection for stale models
- Browsable Obsidian-compatible knowledge vault (`dreamcatcher wiki`)
- Memory linting with rule-based + LLM consistency checks
- White paper and X post
