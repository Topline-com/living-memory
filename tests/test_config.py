"""Tests for dreamcatcher.config.DreamcatcherConfig."""
import yaml
import pytest
from pathlib import Path

from dreamcatcher.config import (
    DreamcatcherConfig, ModelConfig, TrainingConfig,
    ExtractionConfig, ServerConfig,
)


class TestDefaults:
    def test_default_model(self):
        cfg = DreamcatcherConfig()
        assert cfg.model.name == "google/gemma-4-E2B-it"
        assert "Qwen/Qwen3.5-0.8B" in cfg.model.fallbacks

    def test_default_training(self):
        cfg = DreamcatcherConfig()
        assert cfg.training.epochs == 3
        assert cfg.training.batch_size == 4
        assert cfg.training.learning_rate == 5e-6

    def test_default_extraction(self):
        cfg = DreamcatcherConfig()
        assert cfg.extraction.provider == "openrouter"
        assert cfg.extraction.model == "minimax/minimax-m2.7"

    def test_default_server(self):
        cfg = DreamcatcherConfig()
        assert cfg.server.port == 8420
        assert cfg.server.stale_model_threshold_hours == 36.0

    def test_default_paths(self):
        cfg = DreamcatcherConfig()
        assert cfg.db_path == "./data/memory.db"
        assert cfg.sessions_dir == "./data/sessions"


class TestLoadFromYaml:
    def test_load_model_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "model": {"name": "Qwen/Qwen3.5-0.8B", "max_seq_length": 1024},
        }))
        cfg = DreamcatcherConfig.load(str(config_file))
        assert cfg.model.name == "Qwen/Qwen3.5-0.8B"
        assert cfg.model.max_seq_length == 1024

    def test_load_training_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "training": {"epochs": 5, "learning_rate": 2e-5},
        }))
        cfg = DreamcatcherConfig.load(str(config_file))
        assert cfg.training.epochs == 5
        assert cfg.training.learning_rate == 2e-5

    def test_load_extraction_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "extraction": {"provider": "openai", "model": "gpt-4o-mini"},
        }))
        cfg = DreamcatcherConfig.load(str(config_file))
        assert cfg.extraction.provider == "openai"
        assert cfg.extraction.model == "gpt-4o-mini"

    def test_load_paths(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "paths": {"db_path": "/custom/db.sqlite", "sessions_dir": "/custom/sessions"},
        }))
        cfg = DreamcatcherConfig.load(str(config_file))
        assert cfg.db_path == "/custom/db.sqlite"
        assert cfg.sessions_dir == "/custom/sessions"

    def test_load_missing_file_uses_defaults(self, tmp_path):
        cfg = DreamcatcherConfig.load(str(tmp_path / "nonexistent.yaml"))
        assert cfg.model.name == "google/gemma-4-E2B-it"

    def test_unknown_keys_ignored(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "model": {"name": "test", "unknown_key": "value"},
        }))
        cfg = DreamcatcherConfig.load(str(config_file))
        assert cfg.model.name == "test"


class TestEnsureDirs:
    def test_ensure_dirs_creates_directories(self, tmp_path):
        cfg = DreamcatcherConfig()
        cfg.sessions_dir = str(tmp_path / "a" / "b" / "sessions")
        cfg.training_dir = str(tmp_path / "a" / "b" / "training")
        cfg.models_dir = str(tmp_path / "a" / "b" / "models")

        cfg.ensure_dirs()

        assert Path(cfg.sessions_dir).exists()
        assert Path(cfg.training_dir).exists()
        assert Path(cfg.models_dir).exists()

    def test_load_does_not_create_dirs(self, tmp_path):
        """Config.load() should NOT create directories as a side effect."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "paths": {
                "sessions_dir": str(tmp_path / "should_not_exist" / "sessions"),
            },
        }))
        cfg = DreamcatcherConfig.load(str(config_file))
        assert not Path(tmp_path / "should_not_exist" / "sessions").exists()
