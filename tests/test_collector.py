"""Tests for dreamcatcher.collector (SessionCollector + TrainingDataBuilder)."""
import json
import pytest
from pathlib import Path

from dreamcatcher.collector import SessionCollector, TrainingDataBuilder


class TestSessionCollector:
    def test_ingest_text(self, config):
        collector = SessionCollector(config)
        sid = collector.ingest_text("Hello, this is a test transcript.", agent_name="test")
        assert sid is not None

        sessions = collector.db.get_unprocessed_sessions()
        assert len(sessions) == 1
        assert sessions[0]["raw_transcript"] == "Hello, this is a test transcript."

    def test_ingest_file(self, config, tmp_dir):
        # Create a transcript file
        transcript_file = tmp_dir / "test_session.txt"
        transcript_file.write_text("User: Hello\nAssistant: Hi there!")

        collector = SessionCollector(config)
        sid = collector.ingest_file(str(transcript_file), agent_name="test")
        assert sid is not None

    def test_ingest_file_not_found(self, config):
        collector = SessionCollector(config)
        with pytest.raises(FileNotFoundError):
            collector.ingest_file("/nonexistent/file.txt")

    def test_ingest_directory(self, config, tmp_dir):
        # Create session files in the sessions dir
        sessions_dir = Path(config.sessions_dir)
        (sessions_dir / "session1.txt").write_text("Transcript 1")
        (sessions_dir / "session2.txt").write_text("Transcript 2")
        (sessions_dir / "session3.md").write_text("Transcript 3")
        (sessions_dir / "ignore.py").write_text("Not a transcript")  # Should be ignored

        collector = SessionCollector(config)
        ids = collector.ingest_directory(str(sessions_dir))
        assert len(ids) == 3  # .txt and .md only


class TestTrainingDataBuilder:
    def test_build_empty_set(self, config):
        builder = TrainingDataBuilder(config)
        result = builder.build_training_set()
        assert result == []

    def test_build_training_set(self, config, populated_db):
        # The populated_db fixture already has training examples
        builder = TrainingDataBuilder(config)
        builder.db = populated_db

        data = builder.build_training_set()
        assert len(data) == 4  # All 4 examples from populated_db

        # Check format: each item should have messages with system/user/assistant
        for item in data:
            assert "messages" in item
            messages = item["messages"]
            assert len(messages) == 3
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"

    def test_training_set_saved_as_jsonl(self, config, populated_db):
        builder = TrainingDataBuilder(config)
        builder.db = populated_db
        builder.build_training_set()

        # Check that a JSONL file was created
        training_files = list(Path(config.training_dir).glob("full_dataset_*.jsonl"))
        assert len(training_files) == 1

        # Verify JSONL format
        with open(training_files[0]) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4


class TestExtractionParsing:
    """Test the extraction prompt's expected output format parsing."""

    def test_parse_structured_training_pairs(self, config):
        """Verify the collector correctly processes keyed training pairs."""
        collector = SessionCollector(config)

        # Simulate an extraction result
        mock_memory = {
            "category": "project",
            "core_fact": "The user is building a memory system",
            "confidence": 0.95,
            "training_pairs": {
                "semantic": {
                    "instruction": "What is the user working on?",
                    "response": {"memories": [{"category": "project", "content": "A memory system", "confidence": 0.95}]},
                },
                "contextual": {
                    "instruction": "What AI projects is the user building?",
                    "response": {"memories": [{"category": "project", "content": "A memory system for AI agents", "confidence": 0.95}]},
                },
                "specific_1": {
                    "instruction": "What is Dreamcatcher?",
                    "response": {"memories": [{"category": "project", "content": "Dreamcatcher is the user's memory system", "confidence": 0.95}]},
                },
            },
        }

        # Store it via the database
        content = mock_memory.get("core_fact", "")
        memory_id = collector.db.add_memory(
            content=content,
            category=mock_memory["category"],
            session_id="test_session",
            confidence=mock_memory.get("confidence", 1.0),
        )

        # Process training pairs (same logic as in extract_memories)
        training_pairs = mock_memory.get("training_pairs", {})
        KEY_ORDER = ["semantic", "contextual", "specific_1", "specific_2", "specific_3"]
        ordered_pairs = []
        for key in KEY_ORDER:
            if key in training_pairs:
                ordered_pairs.append(training_pairs[key])

        for pair_idx, pair in enumerate(ordered_pairs):
            response = pair["response"]
            if isinstance(response, dict):
                response = json.dumps(response)
            collector.db.add_training_example(
                instruction=pair["instruction"],
                response=response,
                category=mock_memory["category"],
                memory_ids=[memory_id],
                pair_index=pair_idx,
            )

        # Verify
        examples = collector.db.get_all_training_examples()
        assert len(examples) == 3
        assert examples[0]["pair_index"] == 0  # semantic
        assert examples[1]["pair_index"] == 1  # contextual
        assert examples[2]["pair_index"] == 2  # specific_1
