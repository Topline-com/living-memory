"""Tests for team memory functionality."""
import json
import pytest
from pathlib import Path

from dreamcatcher.config import DreamcatcherConfig
from dreamcatcher.database import MemoryDB
from dreamcatcher.teams import TeamMemoryManager, validate_team_id


class TestTeamIdValidation:
    def test_valid_ids(self):
        assert validate_team_id("my-team") == "my-team"
        assert validate_team_id("team_123") == "team_123"
        assert validate_team_id("CompanyABC") == "CompanyABC"
        assert validate_team_id("a") == "a"

    def test_invalid_ids(self):
        with pytest.raises(ValueError):
            validate_team_id("")
        with pytest.raises(ValueError):
            validate_team_id("-starts-with-dash")
        with pytest.raises(ValueError):
            validate_team_id("has spaces")
        with pytest.raises(ValueError):
            validate_team_id("has/slashes")
        with pytest.raises(ValueError):
            validate_team_id("a" * 65)  # Too long


class TestTeamMemoryManager:
    @pytest.fixture
    def manager(self, config):
        return TeamMemoryManager(config)

    def test_list_teams_empty(self, manager):
        assert manager.list_teams() == []

    def test_get_db_creates_team(self, manager):
        db = manager.get_db("test-team")
        assert isinstance(db, MemoryDB)
        assert manager.team_exists("test-team")

    def test_team_isolation(self, manager):
        """Two teams should have completely separate databases."""
        db_a = manager.get_db("team-alpha")
        db_b = manager.get_db("team-beta")

        # Add a memory to team A
        db_a.add_memory("Team Alpha has a project called Falcon",
                        category="project", session_id="s1")

        # Team B should have no memories
        assert len(db_b.get_active_memories()) == 0
        assert len(db_a.get_active_memories()) == 1

    def test_list_teams_after_creation(self, manager):
        manager.get_db("team-one")
        manager.get_db("team-two")
        teams = manager.list_teams()
        assert "team-one" in teams
        assert "team-two" in teams

    def test_team_stats(self, manager):
        db = manager.get_db("stats-team")
        db.add_session("User: Hello\nAssistant: Hi!", agent_name="agent-1")
        db.add_memory("Test fact", category="fact", session_id="s1")

        stats = manager.team_stats("stats-team")
        assert stats["team_id"] == "stats-team"
        assert stats["total_sessions"] == 1
        assert stats["active_memories"] == 1

    def test_get_config_scopes_paths(self, manager):
        cfg = manager.get_config("my-team")
        assert "teams/my-team" in cfg.db_path
        assert "teams/my-team" in cfg.sessions_dir
        assert "teams/my-team" in cfg.training_dir
        assert "teams/my-team" in cfg.models_dir

    def test_get_collector(self, manager):
        """Collector should be wired to the team's database."""
        collector = manager.get_collector("collector-team")
        sid = collector.ingest_text("User: Test transcript\nAssistant: OK",
                                    agent_name="test-agent")
        assert sid is not None

        # Verify it landed in the team's DB
        db = manager.get_db("collector-team")
        sessions = db.get_unprocessed_sessions()
        assert len(sessions) == 1


class TestConfigForTeam:
    def test_for_team_returns_new_config(self, config):
        team_cfg = config.for_team("alpha")
        assert team_cfg is not config
        assert "teams/alpha" in team_cfg.db_path
        assert config.db_path != team_cfg.db_path

    def test_for_team_preserves_non_path_settings(self, config):
        team_cfg = config.for_team("beta")
        assert team_cfg.model.name == config.model.name
        assert team_cfg.training.epochs == config.training.epochs
        assert team_cfg.extraction.provider == config.extraction.provider
        assert team_cfg.server.port == config.server.port


class TestCrossAgentSharing:
    """Test that multiple agents in a team share the same memory pool."""

    @pytest.fixture
    def team_db(self, config):
        manager = TeamMemoryManager(config)
        return manager.get_db("shared-team")

    def test_multiple_agents_share_memories(self, team_db):
        # Agent 1 contributes a session
        team_db.add_session("User: I prefer dark mode\nAssistant: Noted.",
                            agent_name="claude-code", session_id="s1")
        team_db.add_memory("User prefers dark mode",
                           category="preference", session_id="s1")

        # Agent 2 contributes a session
        team_db.add_session("User: Project Alpha deadline is Friday\nAssistant: Got it.",
                            agent_name="hermes", session_id="s2")
        team_db.add_memory("Project Alpha deadline is Friday",
                           category="project", session_id="s2")

        # All memories are visible to any agent querying
        memories = team_db.get_active_memories()
        assert len(memories) == 2
        contents = {m["content"] for m in memories}
        assert "User prefers dark mode" in contents
        assert "Project Alpha deadline is Friday" in contents

    def test_training_set_includes_all_agents(self, team_db):
        # Two agents contribute training examples
        team_db.add_memory("Fact A", category="fact", session_id="s1")
        team_db.add_training_example("Q about A?", "Answer A",
                                      category="fact", pair_index=0)

        team_db.add_memory("Fact B", category="fact", session_id="s2")
        team_db.add_training_example("Q about B?", "Answer B",
                                      category="fact", pair_index=0)

        examples = team_db.get_all_training_examples()
        assert len(examples) == 2
