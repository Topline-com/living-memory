"""Tests for dreamcatcher.server (FastAPI inference server)."""
import json
import pytest
from unittest.mock import patch

from dreamcatcher.server import create_app, _parse_memories


class TestParseMemories:
    """Test JSON parsing of model output."""

    def test_parse_valid_json(self):
        raw = '{"memories": [{"category": "fact", "content": "User is Alex", "confidence": 0.95}]}'
        result = _parse_memories(raw)
        assert len(result) == 1
        assert result[0]["content"] == "User is Alex"

    def test_parse_json_array(self):
        raw = '[{"category": "fact", "content": "Test", "confidence": 1.0}]'
        result = _parse_memories(raw)
        assert len(result) == 1

    def test_parse_json_with_surrounding_text(self):
        raw = 'Here is the response: {"memories": [{"category": "fact", "content": "Test", "confidence": 1.0}]} end.'
        result = _parse_memories(raw)
        assert len(result) == 1

    def test_parse_empty_string(self):
        assert _parse_memories("") == []

    def test_parse_invalid_json(self):
        assert _parse_memories("not json at all") == []

    def test_parse_none(self):
        assert _parse_memories(None) == []


@pytest.fixture
def test_client(config):
    """Create a FastAPI TestClient with proper lifespan handling."""
    from fastapi.testclient import TestClient
    app = create_app(config)
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    def test_health_returns_ok(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False  # No model in test env
        assert "model_age_hours" in data
        assert "stats" in data


class TestIngestEndpoint:
    def test_ingest_stores_session(self, test_client):
        resp = test_client.post("/ingest", json={
            "transcript": "User: Hello\nAssistant: Hi!",
            "agent_name": "test-agent",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "stored"

    def test_ingest_empty_transcript(self, test_client):
        resp = test_client.post("/ingest", json={
            "transcript": "",
            "agent_name": "test",
        })
        # Should still work (empty but valid)
        assert resp.status_code == 200


class TestMemoriesEndpoint:
    def test_list_memories_empty(self, test_client):
        resp = test_client.get("/memories")
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories"] == []
        assert data["count"] == 0


class TestStatsEndpoint:
    def test_stats_returns_data(self, test_client):
        resp = test_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_sessions" in data
