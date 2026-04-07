# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2026.4.7] - 2026-04-07

### Added
- **Team Memory** — Shared memory pools for AI agent teams. Each team gets an isolated SQLite database, nightly training cycle, and fine-tuned model at `data/teams/{team_id}/`.
- **`TeamMemoryManager`** (`dreamcatcher/teams.py`) — Manages per-team databases, collectors, and configs with validated team IDs.
- **`DreamcatcherConfig.for_team()`** — Factory method that returns a config copy with all paths scoped to a team directory.
- **7 new team API endpoints** — `/teams/{id}/ingest`, `/teams/{id}/recall`, `/teams/{id}/context`, `/teams/{id}/memories`, `/teams/{id}/stats`, `/teams/{id}/health`, and `/teams` listing.
- **Team CLI commands** — `dreamcatcher team list`, `dreamcatcher team stats <id>`, `dreamcatcher team nightly [<id>]`.
- **Team client methods** — `team_save_session()`, `team_recall()`, `team_get_context()`, `team_get_memories()` on the `LivingMemory` client class.
- **Paperclip integration** (`integrations/paperclip/`) — TypeScript plugin for Paperclip AI that maps `companyId` to Living Memory `team_id`. Registers 3 agent tools (`recall_team_memory`, `save_to_team_memory`, `team_memory_status`), auto-ingests transcripts on `agent.run.completed`, and schedules nightly training triggers.
- **13 new tests** in `test_teams.py` covering team isolation, cross-agent sharing, config scoping, and collector wiring.

### Fixed
- **OpenRouter default fully wired** — Default extraction provider is now OpenRouter with `openai` as a core dependency. Anthropic available as opt-in (`pip install dreamcatcher-memory[anthropic]`).
- **`lstrip('./')` mangled parent paths** — Replaced with proper `Path` resolution that preserves `../` semantics in config paths.
- **Empty extractions retried forever** — Extraction failures (`None`) now distinguished from successful-empty results (`[]`); boring sessions marked done.
- **`INSERT OR REPLACE` overwrote provenance** — Changed to `INSERT OR IGNORE` + `UPDATE` for memories; added `last_seen_at` column with migration.
- **`all_memories.extend(None)` crash** — Moved `None` check before `extend()` in collector extraction loop.
- **Unauthenticated API exposure** — Server now binds to `127.0.0.1` by default instead of `0.0.0.0`.
- **Superseded memories leaked into training** — Training set now filters out examples whose source memories are all inactive.
- **`/context` dumped all memories** — Replaced unfiltered `get_active_memories(limit=50)` with query-scoped `_search_db()`.
- **Paths with spaces broke setup** — Quoted `project_dir` in bash `-c` command for Claude Code setup.
- **`trust_remote_code=True` unpinned** — Gated on known model families in trainer; removed from local inference in server.

### Changed
- Nightly pipeline (`dreamcatcher nightly`) now auto-runs for all teams after the personal pipeline.

## [0.2.1] - 2026-04-06

### Fixed
- **macOS sandbox compatibility** — MCP server no longer crashes with `PermissionError: Operation not permitted` on macOS. The `mcp` subcommand now skips `ensure_dirs()` since it only talks to the HTTP API and doesn't need local data directories.
- **macOS `com.apple.provenance` blocking** — `dreamcatcher setup claude-code` now generates a `/bin/bash -c` wrapper on macOS, avoiding the sandbox that blocks Python binaries inside project directories.
- **Read-only file system error** — The `cd` in the bash wrapper ensures `config.yaml` and `data/` paths resolve correctly regardless of the app's launch directory.
- **Editable install breaking MCP** on uv-managed Python 3.12 — documented workaround and fix.
- **MLX model loading** for inference server.
- **dotenv loading** and mlx-lm Gemma 4 compatibility in onboarding flow.
- Fixed placeholder `[username]` URLs in Hermes and OpenClaw integration READMEs.

### Changed
- **MCP server renamed** — `"living-memory"` -> `"Living Memory"` in FastMCP constructor, config keys, log prefix, and all documentation.
- **Setup command configures both CLI and Desktop app** — `--global` now writes to both `~/.claude/settings.json` and `claude_desktop_config.json` in one step.
- **Recommend pipx** for Claude Code installation to avoid venv provenance issues on macOS.
- Added Python 3.10+ requirement prominently in README quickstart.
- Session saving guidance now enforces full unfiltered transcripts (no summaries).

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
