# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-06

### Added
- `CLAUDE.md` with session-saving instructions for Claude Code users
- `CHANGELOG.md` for tracking version history
- `CONTRIBUTING.md` with contribution guidelines
- Test suite with pytest (`tests/`)
- GitHub Actions CI (lint + test on every push/PR)
- `ExtractionConfig` dataclass for configurable extraction provider/model
- `model_age_hours` field in `/health` endpoint response
- `config.ensure_dirs()` method (replaces implicit dir creation on load)
- `WikiExporter` and `MemoryLinter` to `__all__` exports in `__init__.py`
- Backward-compatibility aliases `PersonalMemory` and `DreamcatcherMemory` in client
- Multi-stage Dockerfile (separate `server` and `training` targets)

### Changed
- Client class renamed from `PersonalMemory` to `LivingMemory`
- `collector.py` now reads extraction model from config instead of hardcoding
- `DreamcatcherConfig.load()` no longer creates directories as a side effect
- Docker Compose uses build targets for lighter server image

### Fixed
- `pyproject.toml` placeholder URLs and author name
- `docs/x-post.md` placeholder links
- `build/` and `egg-info/` directories removed from git tracking
- `.gitignore` now covers `.venv/` and `dreamcatcher_memory.egg-info/`

## [0.4.0] - 2026-04-06

### Added
- Memory linting system (`dreamcatcher lint`) with rule-based + LLM passes
- Browsable Obsidian-compatible knowledge vault (`dreamcatcher wiki`)
- Wiki sync-back: edit YAML frontmatter in vault to deprecate/delete memories
- Organic reinforcement density indicators in vault export

## [0.3.0] - 2026-04-06

### Added
- Semantic compression for training set rendering
- `pair_index` field on training examples for generality ordering
- Compression stats in `dreamcatcher stats` output

## [0.2.0] - 2026-04-06

### Added
- MCP server for Claude Code integration
- Dual-backend trainer (MLX on Apple Silicon, PyTorch on NVIDIA)
- Disaster recovery delta context injection for stale models
- Circuit breaker in MCP server

## [0.1.0] - 2026-04-06

### Added
- Initial release
- Core architecture: sessions, memory extraction, training, inference
- FastAPI inference server
- CLI entry points
- Docker support
- Single-file client library
