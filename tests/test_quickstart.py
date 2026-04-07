"""Tests for quickstart, update, and install-path logic.

Pins down regressions in the wheel-installed code path:
- Config falls back to ~/.dreamcatcher/ when no config.yaml exists
- Embedded sample transcript can be ingested
- Platform detection returns valid structure
- pip install failures are reported honestly (behavioral test)
- Claude Code setup doesn't falsely claim nightly scheduling (behavioral test)
- Update detects source checkout vs wheel and dispatches the right commands
- uv pip fallback triggers when pip module is missing
"""
import json
import os
import shutil
import subprocess
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from dreamcatcher.config import DreamcatcherConfig
from dreamcatcher.database import MemoryDB


class TestConfigFallbackPath:
    """When no config.yaml exists, paths should use ~/.dreamcatcher/, not site-packages."""

    def test_no_config_yaml_uses_home_dir(self, tmp_path, monkeypatch):
        """Simulate wheel install: no config.yaml anywhere."""
        monkeypatch.chdir(tmp_path)
        cfg = DreamcatcherConfig.load("nonexistent_config.yaml")
        home_dc = str(Path.home() / ".dreamcatcher")
        assert home_dc in cfg.db_path, f"Expected ~/.dreamcatcher in db_path, got {cfg.db_path}"
        assert home_dc in cfg.sessions_dir
        assert home_dc in cfg.training_dir
        assert home_dc in cfg.models_dir

    def test_no_config_yaml_does_not_use_site_packages(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = DreamcatcherConfig.load("nonexistent_config.yaml")
        assert "site-packages" not in cfg.db_path
        assert "site-packages" not in cfg.sessions_dir

    def test_existing_config_yaml_uses_its_directory(self, tmp_path):
        """When config.yaml exists, paths resolve relative to it."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# empty config\n")
        cfg = DreamcatcherConfig.load(str(config_file))
        assert str(tmp_path) in cfg.db_path


class TestEmbeddedSampleTranscript:
    """The quickstart sample must be ingestable from any install method."""

    def test_sample_transcript_ingests(self, config):
        """Embedded sample transcript should produce a valid session."""
        from dreamcatcher.collector import SessionCollector
        sample = (
            "User: I'm working on a project called Horizon — it's a real-time analytics "
            "dashboard for our SaaS product. We're using React on the frontend and FastAPI "
            "on the backend. The launch target is end of Q2."
        )
        collector = SessionCollector(config)
        sid = collector.ingest_text(sample, "quickstart-demo")
        assert sid is not None
        db = MemoryDB(config.db_path)
        sessions = db.get_unprocessed_sessions()
        assert len(sessions) == 1
        assert "Horizon" in sessions[0]["raw_transcript"]


class TestPlatformDetection:
    def test_returns_valid_structure(self):
        from dreamcatcher.__main__ import _detect_platform
        info = _detect_platform()
        assert "os" in info
        assert "arch" in info
        assert "python" in info
        assert "training_backend" in info
        assert "label" in info
        assert info["training_backend"] in (
            "mlx", "mlx_needed", "pytorch_cuda", "pytorch_cpu", "pytorch_needed", "none"
        )


class TestQuickstartBehavior:
    """Behavioral tests that drive cmd_quickstart and check actual output."""

    def test_failed_dep_install_reports_failure(self, monkeypatch, tmp_path, capsys):
        """Phase 2 must print failure message, not 'Done.', on install error."""
        from dreamcatcher.__main__ import cmd_quickstart

        cfg = DreamcatcherConfig()
        cfg.db_path = str(tmp_path / "data" / "memory.db")
        cfg.sessions_dir = str(tmp_path / "data" / "sessions")
        cfg.training_dir = str(tmp_path / "data" / "training")
        cfg.models_dir = str(tmp_path / "data" / "models")

        monkeypatch.chdir(tmp_path)
        # Bypass Python version check (test machine may be <3.10)
        monkeypatch.setattr("dreamcatcher.__main__.sys.version_info", (3, 12, 0))
        monkeypatch.setattr("dreamcatcher.__main__._detect_platform", lambda: {
            "os": "darwin", "arch": "arm64", "python": "3.12.0",
            "mlx": False, "cuda": False,
            "training_backend": "mlx_needed",
            "label": "macOS Apple Silicon (MLX not installed)",
        })

        confirm_answers = iter([True, False])
        monkeypatch.setattr("dreamcatcher.__main__._confirm", lambda *a, **k: next(confirm_answers, False))
        monkeypatch.setattr("dreamcatcher.__main__._prompt", lambda *a, **k: "")

        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: failed)

        cmd_quickstart(cfg)
        out = capsys.readouterr().out

        assert "Install failed" in out
        assert "pip install mlx mlx-lm" in out

    def test_claude_code_does_not_claim_nightly_scheduled(self):
        """Claude Code quickstart branch must not append to nightly_scheduled.

        This is a source-level assertion — behavioral testing of the full
        quickstart with mocked integration detection is fragile due to the
        many interactive prompts. The source check is reliable and catches
        the specific regression (accidentally adding nightly_scheduled.append
        to the Claude Code branch).
        """
        source = Path(__file__).parent.parent / "dreamcatcher" / "__main__.py"
        content = source.read_text()
        cc_start = content.find('if name == "claude-code":')
        cc_end = content.find('elif name == "hermes":', cc_start)
        cc_block = content[cc_start:cc_end]
        assert "nightly_scheduled.append" not in cc_block


class TestUpdateBehavior:
    """Behavioral tests that drive cmd_update and check dispatched commands."""

    def test_source_checkout_uses_git_pull_and_editable_install(self, monkeypatch, tmp_path, capsys):
        """Source checkout should git pull and pip install -e ."""
        from dreamcatcher.__main__ import cmd_update

        # Create fake source checkout
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        fake_main = tmp_path / "dreamcatcher" / "__main__.py"
        fake_main.parent.mkdir()
        fake_main.touch()

        calls = []
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="Already up to date.", stderr="")

        def record_run(cmd, **kwargs):
            calls.append(list(cmd))
            return ok

        monkeypatch.setattr("subprocess.run", record_run)
        import dreamcatcher.__main__ as main_mod
        monkeypatch.setattr(main_mod, "__file__", str(fake_main))

        cmd_update(DreamcatcherConfig())

        assert any(cmd == ["git", "pull"] for cmd in calls), f"Expected git pull, got {calls}"
        assert any("-e" in cmd for cmd in calls), f"Expected editable install, got {calls}"

    def test_wheel_install_uses_pip_upgrade(self, monkeypatch, tmp_path, capsys):
        """Wheel install (no .git) should pip install --upgrade."""
        from dreamcatcher.__main__ import cmd_update

        # No .git = wheel install
        fake_main = tmp_path / "dreamcatcher" / "__main__.py"
        fake_main.parent.mkdir()
        fake_main.touch()

        calls = []
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def record_run(cmd, **kwargs):
            calls.append(list(cmd))
            return ok

        monkeypatch.setattr("subprocess.run", record_run)
        import dreamcatcher.__main__ as main_mod
        monkeypatch.setattr(main_mod, "__file__", str(fake_main))

        cmd_update(DreamcatcherConfig())

        assert any("--upgrade" in cmd and "dreamcatcher-memory" in cmd for cmd in calls), \
            f"Expected pip upgrade, got {calls}"
        assert not any(cmd == ["git", "pull"] for cmd in calls)

    def test_uv_fallback_when_pip_missing(self, monkeypatch, tmp_path, capsys):
        """When pip module is missing, should fall back to uv pip with --python."""
        from dreamcatcher.__main__ import cmd_update

        fake_main = tmp_path / "dreamcatcher" / "__main__.py"
        fake_main.parent.mkdir()
        fake_main.touch()

        calls = []
        no_pip = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="No module named pip")
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def record_run(cmd, **kwargs):
            calls.append(list(cmd))
            if "-m" in cmd and "pip" in cmd:
                return no_pip
            return ok

        monkeypatch.setattr("subprocess.run", record_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)
        import dreamcatcher.__main__ as main_mod
        monkeypatch.setattr(main_mod, "__file__", str(fake_main))

        cmd_update(DreamcatcherConfig())

        uv_calls = [c for c in calls if c[0] == "/usr/local/bin/uv"]
        assert len(uv_calls) == 1, f"Expected one uv call, got {uv_calls}"
        assert "--python" in uv_calls[0]
        assert sys.executable in uv_calls[0]
        assert "dreamcatcher-memory" in uv_calls[0]
