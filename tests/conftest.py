"""Shared test fixtures for Dreamcatcher tests."""
import os
import pytest
import tempfile
from pathlib import Path

from dreamcatcher.config import DreamcatcherConfig
from dreamcatcher.database import MemoryDB


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def config(tmp_dir):
    """Create a DreamcatcherConfig pointing at temporary directories."""
    cfg = DreamcatcherConfig()
    cfg.db_path = str(tmp_dir / "test_memory.db")
    cfg.sessions_dir = str(tmp_dir / "sessions")
    cfg.training_dir = str(tmp_dir / "training")
    cfg.models_dir = str(tmp_dir / "models")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def db(config):
    """Create a fresh MemoryDB instance for testing."""
    return MemoryDB(config.db_path)


@pytest.fixture
def populated_db(db):
    """A MemoryDB pre-loaded with sample data for testing."""
    # Add sessions
    db.add_session("User: What's the status of Project Alpha?\nAssistant: Project Alpha is on track.",
                   agent_name="test-agent", session_id="session_001")
    db.add_session("User: My wife Shannon and I are planning a trip.\nAssistant: That sounds great!",
                   agent_name="test-agent", session_id="session_002")

    # Add memories
    db.add_memory("The user is working on Project Alpha, a software platform",
                  category="project", session_id="session_001", confidence=0.95)
    db.add_memory("The user's wife is named Shannon",
                  category="relationship", session_id="session_002", confidence=1.0)
    db.add_memory("The user prefers direct, concise communication",
                  category="preference", session_id="session_001", confidence=0.85)
    db.add_memory("The user has two sons named Teddy and Michael",
                  category="relationship", session_id="session_002", confidence=1.0)

    # Add training examples with varying pair_index
    db.add_training_example(
        instruction="What projects is the user working on?",
        response='{"memories": [{"category": "project", "content": "Project Alpha", "confidence": 0.95}]}',
        category="project", memory_ids=["mem_001"], pair_index=0,
    )
    db.add_training_example(
        instruction="What is Project Alpha?",
        response='{"memories": [{"category": "project", "content": "Project Alpha is a software platform", "confidence": 0.95}]}',
        category="project", memory_ids=["mem_001"], pair_index=1,
    )
    db.add_training_example(
        instruction="What is the status of Project Alpha?",
        response='{"memories": [{"category": "project", "content": "Project Alpha is on track", "confidence": 0.9}]}',
        category="project", memory_ids=["mem_001"], pair_index=2,
    )
    db.add_training_example(
        instruction="Who is the user's wife?",
        response='{"memories": [{"category": "relationship", "content": "Shannon", "confidence": 1.0}]}',
        category="relationship", memory_ids=["mem_002"], pair_index=0,
    )

    return db
