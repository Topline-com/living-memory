"""Tests for dreamcatcher.database.MemoryDB."""
import json
import pytest
from datetime import datetime, timezone, timedelta

from dreamcatcher.database import MemoryDB


class TestSessions:
    def test_add_session(self, db):
        sid = db.add_session("Hello world transcript", agent_name="test")
        assert sid is not None
        assert len(sid) == 16  # SHA256 hex prefix

    def test_add_session_with_custom_id(self, db):
        sid = db.add_session("Transcript text", session_id="custom_123")
        assert sid == "custom_123"

    def test_get_unprocessed_sessions(self, db):
        db.add_session("Session 1", session_id="s1")
        db.add_session("Session 2", session_id="s2")

        unprocessed = db.get_unprocessed_sessions()
        assert len(unprocessed) == 2
        assert unprocessed[0]["id"] == "s1"
        assert unprocessed[1]["id"] == "s2"

    def test_mark_session_processed(self, db):
        db.add_session("Session 1", session_id="s1")
        db.mark_session_processed("s1")

        unprocessed = db.get_unprocessed_sessions()
        assert len(unprocessed) == 0

    def test_session_stores_agent_name(self, db):
        db.add_session("Transcript", agent_name="hermes", session_id="s1")
        unprocessed = db.get_unprocessed_sessions()
        assert unprocessed[0]["agent_name"] == "hermes"

    def test_session_token_count_estimated(self, db):
        text = "word " * 100  # 100 words
        db.add_session(text, session_id="s1")
        sessions = db.get_unprocessed_sessions()
        assert sessions[0]["token_count"] == 130  # ~1.3x word count


class TestMemories:
    def test_add_memory(self, db):
        db.add_session("transcript", session_id="s1")
        mid = db.add_memory("User likes Python", category="preference",
                            session_id="s1", confidence=0.9)
        assert mid is not None

    def test_get_active_memories(self, db):
        db.add_session("transcript", session_id="s1")
        db.add_memory("Fact 1", category="fact", session_id="s1")
        db.add_memory("Fact 2", category="project", session_id="s1")

        memories = db.get_active_memories()
        assert len(memories) == 2

    def test_filter_by_category(self, db):
        db.add_session("transcript", session_id="s1")
        db.add_memory("Fact 1", category="fact", session_id="s1")
        db.add_memory("Pref 1", category="preference", session_id="s1")

        facts = db.get_active_memories(category="fact")
        assert len(facts) == 1
        assert facts[0]["category"] == "fact"

    def test_supersede_memory(self, db):
        db.add_session("transcript", session_id="s1")
        m1 = db.add_memory("Old fact", category="fact", session_id="s1")
        m2 = db.add_memory("New fact", category="fact", session_id="s1")
        db.supersede_memory(m1, m2)

        active = db.get_active_memories()
        assert len(active) == 1
        assert active[0]["content"] == "New fact"

    def test_memory_confidence(self, db):
        db.add_session("transcript", session_id="s1")
        db.add_memory("Confident fact", category="fact", session_id="s1",
                       confidence=0.95)
        memories = db.get_active_memories()
        assert memories[0]["confidence"] == 0.95


class TestTrainingExamples:
    def test_add_training_example(self, db):
        eid = db.add_training_example(
            instruction="What is the user's name?",
            response='{"memories": []}',
            category="fact",
            pair_index=0,
        )
        assert eid is not None

    def test_get_all_training_examples(self, db):
        db.add_training_example("Q1", "A1", "fact", pair_index=0)
        db.add_training_example("Q2", "A2", "fact", pair_index=1)

        examples = db.get_all_training_examples()
        assert len(examples) == 2

    def test_pair_index_stored(self, db):
        db.add_training_example("Q1", "A1", "fact", pair_index=3)
        examples = db.get_all_training_examples()
        assert examples[0]["pair_index"] == 3

    def test_memory_ids_stored_as_json(self, db):
        db.add_training_example("Q1", "A1", "fact",
                                memory_ids=["m1", "m2"], pair_index=0)
        examples = db.get_all_training_examples()
        ids = json.loads(examples[0]["memory_ids"])
        assert ids == ["m1", "m2"]

    def test_example_count(self, db):
        assert db.get_training_example_count() == 0
        db.add_training_example("Q1", "A1", "fact")
        assert db.get_training_example_count() == 1


class TestSemanticCompression:
    def test_all_recent_examples_included(self, populated_db):
        """Recent examples should all be included regardless of pair_index."""
        result = populated_db.get_training_set_with_compression(
            compression_age_days=180, max_pair_index_old=1
        )
        # All 4 examples are recent (just created), so all should be included
        assert len(result["examples"]) == 4
        assert result["n_dropped"] == 0

    def test_old_examples_compressed(self, db):
        """Old examples with high pair_index should be excluded."""
        # Manually insert old examples by backdating them
        from datetime import timedelta
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()

        with db._conn() as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO training_examples (id, memory_ids, instruction, response, category, pair_index, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (f"old_{i}", "[]", f"Question {i}", f"Answer {i}", "fact", i, old_date)
                )

        result = db.get_training_set_with_compression(
            compression_age_days=180, max_pair_index_old=1
        )
        # Only pair_index 0 and 1 should survive compression
        assert len(result["examples"]) == 2
        assert result["n_compressed"] == 2
        assert result["n_dropped"] == 3


class TestStats:
    def test_stats_empty_db(self, db):
        stats = db.stats()
        assert stats["total_sessions"] == 0
        assert stats["active_memories"] == 0
        assert stats["total_training_examples"] == 0
        assert stats["training_runs"] == 0

    def test_stats_populated(self, populated_db):
        stats = populated_db.stats()
        assert stats["total_sessions"] == 2
        assert stats["active_memories"] == 4
        assert stats["total_training_examples"] == 4
        assert stats["memories_by_category"]["project"] == 1
        assert stats["memories_by_category"]["relationship"] == 2
        assert stats["memories_by_category"]["preference"] == 1

    def test_log_training_run(self, db):
        run_id = db.log_training_run(
            model_path="/models/memory_20260406",
            num_examples=100,
            loss_final=0.234,
            duration_seconds=45.5,
            model_name="google/gemma-4-E2B-it",
        )
        assert run_id is not None
        stats = db.stats()
        assert stats["training_runs"] == 1
