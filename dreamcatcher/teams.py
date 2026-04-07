"""
Team Memory Manager — Shared Memory Across Agent Teams
=======================================================
Manages per-team SQLite databases at data/teams/{team_id}/memory.db.
Each team (mapped 1:1 to a Paperclip company) gets its own isolated
memory store, nightly training cycle, and fine-tuned model.

All sessions ingested to a team are shared across all agents in the team.
"""
import re
from pathlib import Path
from typing import Optional

from .config import DreamcatcherConfig
from .database import MemoryDB
from .collector import SessionCollector


# Only allow safe team IDs (alphanumeric, hyphens, underscores)
_TEAM_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def validate_team_id(team_id: str) -> str:
    """Validate and return a safe team ID, or raise ValueError."""
    if not _TEAM_ID_RE.match(team_id):
        raise ValueError(
            f"Invalid team_id '{team_id}': must be 1-64 chars, "
            "alphanumeric/hyphens/underscores, starting with alphanumeric"
        )
    return team_id


class TeamMemoryManager:
    """Manages per-team memory databases and services."""

    def __init__(self, base_config: DreamcatcherConfig):
        self._base_config = base_config
        # Teams live under data/teams/ relative to the main data directory
        self._teams_dir = Path(base_config.db_path).parent / "teams"
        self._dbs: dict[str, MemoryDB] = {}
        self._collectors: dict[str, SessionCollector] = {}

    @property
    def teams_dir(self) -> Path:
        return self._teams_dir

    def get_config(self, team_id: str) -> DreamcatcherConfig:
        """Return a config with all paths scoped to a team directory."""
        validate_team_id(team_id)
        return self._base_config.for_team(team_id)

    def get_db(self, team_id: str) -> MemoryDB:
        """Return (or create) the MemoryDB for a team."""
        validate_team_id(team_id)
        if team_id not in self._dbs:
            cfg = self.get_config(team_id)
            cfg.ensure_dirs()
            self._dbs[team_id] = MemoryDB(cfg.db_path)
        return self._dbs[team_id]

    def get_collector(self, team_id: str) -> SessionCollector:
        """Return a SessionCollector wired to a team's DB."""
        validate_team_id(team_id)
        if team_id not in self._collectors:
            cfg = self.get_config(team_id)
            cfg.ensure_dirs()
            self._collectors[team_id] = SessionCollector(cfg)
        return self._collectors[team_id]

    def list_teams(self) -> list[str]:
        """List team IDs that have data directories."""
        if not self._teams_dir.exists():
            return []
        return sorted(
            d.name for d in self._teams_dir.iterdir()
            if d.is_dir() and (d / "memory.db").exists()
        )

    def team_exists(self, team_id: str) -> bool:
        validate_team_id(team_id)
        return (self._teams_dir / team_id / "memory.db").exists()

    def team_stats(self, team_id: str) -> dict:
        """Return stats for a specific team."""
        db = self.get_db(team_id)
        stats = db.stats()
        stats["team_id"] = team_id
        return stats
