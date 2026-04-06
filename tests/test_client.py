"""Tests for dreamcatcher_client.py (LivingMemory client)."""
import pytest
import sys
from pathlib import Path

# Add the repo root to sys.path so we can import the single-file client
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcatcher_client import LivingMemory, PersonalMemory, DreamcatcherMemory


class TestClientClass:
    def test_living_memory_exists(self):
        """LivingMemory should be the primary class."""
        assert LivingMemory is not None

    def test_backward_compat_aliases(self):
        """PersonalMemory and DreamcatcherMemory should alias LivingMemory."""
        assert PersonalMemory is LivingMemory
        assert DreamcatcherMemory is LivingMemory

    def test_default_url(self):
        client = LivingMemory()
        assert client.base_url == "http://localhost:8420"
        client.close()

    def test_custom_url(self):
        client = LivingMemory(base_url="http://myserver:9999")
        assert client.base_url == "http://myserver:9999"
        client.close()

    def test_url_trailing_slash_stripped(self):
        client = LivingMemory(base_url="http://localhost:8420/")
        assert client.base_url == "http://localhost:8420"
        client.close()

    def test_context_manager(self):
        with LivingMemory() as m:
            assert m is not None
            assert m.base_url == "http://localhost:8420"

    def test_is_available_returns_false_when_offline(self):
        """Should gracefully return False when server isn't running."""
        with LivingMemory(base_url="http://localhost:19999") as m:
            assert m.is_available() is False

    def test_get_context_returns_empty_when_offline(self):
        with LivingMemory(base_url="http://localhost:19999") as m:
            assert m.get_context("test") == ""

    def test_get_memories_returns_empty_when_offline(self):
        with LivingMemory(base_url="http://localhost:19999") as m:
            assert m.get_memories("test") == []

    def test_recall_returns_empty_when_offline(self):
        with LivingMemory(base_url="http://localhost:19999") as m:
            assert m.recall("test") == ""

    def test_save_session_returns_none_when_offline(self):
        with LivingMemory(base_url="http://localhost:19999") as m:
            assert m.save_session("transcript") is None
