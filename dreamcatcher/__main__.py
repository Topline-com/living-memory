#!/usr/bin/env python3
"""
Dreamcatcher — Living Memory for AI Agents
=============================================
Usage:
  dreamcatcher quickstart                   Interactive setup wizard (recommended for new users)
  dreamcatcher init                        Initialize directories + download base model
  dreamcatcher ingest <path> [agent]       Ingest session transcripts
  dreamcatcher extract                     Extract memories (frontier LLM API call)
  dreamcatcher build                       Build the nightly training dataset
  dreamcatcher train [--force]             Re-fine-tune the memory model from base weights
  dreamcatcher nightly                     Full pipeline: extract → build → train → benchmark
  dreamcatcher serve                       Start the inference server
  dreamcatcher mcp                         Start the MCP server (for Claude Code)
  dreamcatcher setup claude-code           Configure Claude Code integration
  dreamcatcher wiki                        Export memory vault as browsable markdown
  dreamcatcher wiki --sync                 Sync vault edits back to canonical store
  dreamcatcher lint                        Run memory consistency check (rule-based + LLM)
  dreamcatcher stats                       Show statistics
  dreamcatcher export                      Export memories as JSON
  dreamcatcher cleanup [--keep N]          Remove old model checkpoints
  dreamcatcher update                      Pull latest code + reinstall dependencies
  dreamcatcher uninstall                   Remove configs, integrations, and optionally data
  dreamcatcher reinstall                   Full uninstall + quickstart in one command
  dreamcatcher team list                   List all teams
  dreamcatcher team stats <team_id>        Show team memory stats
  dreamcatcher team nightly [<team_id>]    Run nightly pipeline for one/all teams
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone

# Load .env before anything reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; users can export vars manually

from .config import DreamcatcherConfig
from .database import MemoryDB
from .collector import SessionCollector, TrainingDataBuilder
from .trainer import MemoryTrainer


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    config = DreamcatcherConfig.load()
    # MCP server only talks to the HTTP API — skip creating data dirs
    # (avoids read-only filesystem errors when launched by Claude Desktop)
    if command not in ("mcp", "quickstart", "uninstall", "reinstall"):
        config.ensure_dirs()
    commands = {
        "quickstart": cmd_quickstart,
        "init": cmd_init,
        "ingest": cmd_ingest,
        "extract": cmd_extract,
        "build": cmd_build,
        "train": cmd_train,
        "nightly": cmd_nightly,
        "serve": cmd_serve,
        "mcp": cmd_mcp,
        "setup": cmd_setup,
        "wiki": cmd_wiki,
        "lint": cmd_lint,
        "stats": cmd_stats,
        "export": cmd_export,
        "cleanup": cmd_cleanup,
        "update": cmd_update,
        "uninstall": cmd_uninstall,
        "reinstall": cmd_reinstall,
        "team": cmd_team,
    }

    if command in commands:
        commands[command](config)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


def _prompt(question: str, default: str = "") -> str:
    """Prompt user for input with optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {question}{suffix}: ").strip()
        return answer if answer else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def _confirm(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no with default."""
    hint = "Y/n" if default else "y/N"
    try:
        answer = input(f"  {question} [{hint}]: ").strip().lower()
        if not answer:
            return default
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def _detect_platform() -> dict:
    """Detect OS, architecture, and available training backends."""
    import platform
    import shutil

    info = {
        "os": sys.platform,
        "arch": platform.machine(),
        "python": platform.python_version(),
        "mlx": False,
        "cuda": False,
        "label": "Unknown",
        "training_backend": "none",
    }

    # Check for Apple Silicon + MLX
    if sys.platform == "darwin" and info["arch"] == "arm64":
        info["label"] = "macOS Apple Silicon"
        try:
            import mlx
            info["mlx"] = True
            info["training_backend"] = "mlx"
            info["label"] += " (MLX available)"
        except ImportError:
            info["training_backend"] = "mlx_needed"
            info["label"] += " (MLX not installed)"
    elif sys.platform == "darwin":
        info["label"] = "macOS Intel"
        info["training_backend"] = "pytorch_cpu"
    else:
        info["label"] = "Linux"
        # Check NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                info["cuda"] = True
                info["training_backend"] = "pytorch_cuda"
                info["label"] = f"Linux (NVIDIA {gpu_name})"
            else:
                info["training_backend"] = "pytorch_cpu"
        except ImportError:
            info["training_backend"] = "pytorch_needed"

    return info


def cmd_quickstart(config):
    """Interactive setup wizard — gets everything running in one command."""
    import shutil
    import subprocess

    print(f"\n{'='*60}")
    print(f"  Living Memory — Quickstart Setup")
    print(f"{'='*60}")

    integrations_configured = []
    nightly_scheduled = []

    # ── Phase 1: Platform Detection ────────────────────────────

    print(f"\n  Phase 1: Detecting platform...")
    platform_info = _detect_platform()
    print(f"  Platform: {platform_info['label']}")
    print(f"  Python:   {platform_info['python']}")

    # Python version check
    if sys.version_info < (3, 10):
        print(f"\n  Python 3.10+ required (you have {platform_info['python']}).")
        print(f"  Install with: brew install python@3.12")
        sys.exit(1)

    # ── Phase 2: Training Dependencies ─────────────────────────

    print(f"\n  Phase 2: Training dependencies")
    backend = platform_info["training_backend"]

    def _pip_install(packages: list[str]) -> bool:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + packages,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  Done.")
            return True
        else:
            print(f"  Install failed. Run manually: pip install {' '.join(packages)}")
            return False

    if backend == "mlx":
        print(f"  MLX is installed and ready for training.")
    elif backend == "mlx_needed":
        if _confirm("Install MLX for Apple Silicon training?"):
            print(f"  Installing mlx and mlx-lm...")
            _pip_install(["mlx", "mlx-lm"])
        else:
            print(f"  Skipped. You can install later: pip install mlx mlx-lm")
    elif backend == "pytorch_cuda":
        print(f"  NVIDIA GPU detected. PyTorch with CUDA is ready.")
    elif backend == "pytorch_needed":
        if _confirm("Install PyTorch for GPU/CPU training?"):
            print(f"  Installing training dependencies...")
            _pip_install(["torch>=2.2.0", "transformers>=4.51.0", "datasets>=2.19.0",
                          "accelerate>=0.28.0", "safetensors>=0.4.0"])
        else:
            print(f"  Skipped. Install later: pip install dreamcatcher-memory[train]")
    elif backend == "pytorch_cpu":
        print(f"  No GPU detected. Training will use CPU (slower).")
        if _confirm("Install PyTorch for CPU training?"):
            _pip_install(["torch>=2.2.0", "transformers>=4.51.0", "datasets>=2.19.0",
                          "accelerate>=0.28.0"])

    # ── Phase 3: API Key ───────────────────────────────────────

    print(f"\n  Phase 3: Extraction API key")
    env_path = Path(config.db_path).parent.parent / ".env"
    existing_key = os.environ.get("OPENROUTER_API_KEY", "")

    if existing_key and existing_key != "sk-or-...":
        print(f"  OpenRouter API key already configured.")
    else:
        print(f"  Memory extraction requires an OpenRouter API key (~$0.01-0.05/night).")
        print(f"  Get one at: https://openrouter.ai/keys")
        key = _prompt("OpenRouter API key (or Enter to skip)")
        if key and key != "sk-or-...":
            # Validate the key
            print(f"  Validating key...")
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=key,
                )
                client.models.list()
                print(f"  Key is valid.")
                os.environ["OPENROUTER_API_KEY"] = key
            except Exception:
                print(f"  Could not validate key (may still work). Saving anyway.")

            # Write to .env
            env_lines = []
            if env_path.exists():
                env_lines = env_path.read_text().splitlines()
            # Replace or append
            found = False
            for i, line in enumerate(env_lines):
                if line.startswith("OPENROUTER_API_KEY="):
                    env_lines[i] = f"OPENROUTER_API_KEY={key}"
                    found = True
                    break
            if not found:
                env_lines.append(f"OPENROUTER_API_KEY={key}")
            env_path.write_text("\n".join(env_lines) + "\n")
            print(f"  Saved to {env_path}")
        else:
            print(f"  Skipped. Add your key to .env later.")

    # ── Phase 4: Initialize ────────────────────────────────────

    print(f"\n  Phase 4: Initializing...")
    config.ensure_dirs()
    db = MemoryDB(config.db_path)
    print(f"  Database: {config.db_path}")
    print(f"  Directories created.")

    # ── Phase 5: Integration Detection + Setup ─────────────────

    print(f"\n  Phase 5: Detecting integrations...")
    detected = {}

    # Claude Code
    claude_dir = Path.home() / ".claude"
    if claude_dir.exists():
        detected["claude-code"] = "~/.claude/ directory found"

    # Hermes
    hermes_dir = Path.home() / ".hermes"
    hermes_bin = shutil.which("hermes")
    if hermes_dir.exists() or hermes_bin:
        detected["hermes"] = "Hermes agent detected"

    # OpenClaw
    openclaw_dir = Path.home() / ".openclaw"
    openclaw_bin = shutil.which("openclaw")
    if openclaw_dir.exists() or openclaw_bin:
        detected["openclaw"] = "OpenClaw detected"

    # Paperclip (check if server is running)
    try:
        import httpx
        resp = httpx.get("http://localhost:3000/health", timeout=2.0)
        if resp.status_code == 200:
            detected["paperclip"] = "Paperclip server running on :3000"
    except Exception:
        pass

    if not detected:
        print(f"  No agent platforms detected.")
    else:
        for name, reason in detected.items():
            print(f"  Found: {name} ({reason})")

        for name in detected:
            display = {"claude-code": "Claude Code", "hermes": "Hermes",
                       "openclaw": "OpenClaw", "paperclip": "Paperclip"}[name]
            if _confirm(f"Configure {display} integration?"):
                try:
                    if name == "claude-code":
                        # Inject --global flag for setup
                        sys.argv = ["dreamcatcher", "setup", "claude-code", "--global"]
                        _setup_claude_code(config)
                        integrations_configured.append("Claude Code (MCP)")
                        # Note: nightly scheduling for Claude Code requires
                        # the scheduled task to be activated from within Claude Code
                        # (via /scheduled or the Scheduled tab). MCP setup alone
                        # doesn't create the scheduler.
                    elif name == "hermes":
                        sys.argv = ["dreamcatcher", "setup", "hermes"]
                        _setup_hermes(config)
                        integrations_configured.append("Hermes (plugin)")
                        nightly_scheduled.append("Hermes (cron job)")
                    elif name == "openclaw":
                        sys.argv = ["dreamcatcher", "setup", "openclaw"]
                        _setup_openclaw(config)
                        integrations_configured.append("OpenClaw (plugin)")
                        nightly_scheduled.append("OpenClaw (cron job)")
                    elif name == "paperclip":
                        print(f"  Paperclip requires --company and --agent IDs.")
                        company = _prompt("Paperclip company ID")
                        agent = _prompt("Paperclip agent ID")
                        if company and agent:
                            sys.argv = ["dreamcatcher", "setup", "paperclip",
                                        "--company", company, "--agent", agent]
                            _setup_paperclip(config)
                            integrations_configured.append("Paperclip (routine)")
                            nightly_scheduled.append("Paperclip (scheduled routine)")
                except Exception as e:
                    print(f"  Setup failed: {e}")

    # ── Phase 6: Demo / Sample Transcript ──────────────────────

    print(f"\n  Phase 6: First data")
    sample_path = Path(__file__).parent.parent / "examples" / "sample_transcript.txt"
    if sample_path.exists():
        if _confirm("Ingest a sample transcript to test the system?"):
            collector = SessionCollector(config)
            sid = collector.ingest_file(str(sample_path), "quickstart-demo")
            print(f"  Ingested sample session: {sid}")
            print(f"  Run 'dreamcatcher nightly' to extract memories from it.")
    else:
        print(f"  No sample transcript found. Ingest your own with:")
        print(f"    dreamcatcher ingest <file>")

    # ── Summary ────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Living Memory — Setup Complete!")
    print(f"  {'─'*40}")
    print(f"  Platform:     {platform_info['label']}")
    print(f"  Provider:     OpenRouter ({config.extraction.model})")
    print(f"  Server:       http://{config.server.host}:{config.server.port}")
    if integrations_configured:
        print(f"  Integrations: {', '.join(integrations_configured)}")
    if nightly_scheduled:
        print(f"  Nightly:      {', '.join(nightly_scheduled)}")
    print(f"\n  Next steps:")
    print(f"    1. dreamcatcher serve          Start the inference server")
    print(f"    2. dreamcatcher nightly        Run your first training pipeline")
    print(f"{'='*60}\n")


def cmd_init(config):
    """Initialize Dreamcatcher: create directories, database, and download the base model."""
    print(f"\n  Dreamcatcher — Initializing")
    print(f"  {'─'*40}")

    # Create data directories
    for d in [config.sessions_dir, config.training_dir, config.models_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")

    # Initialize the database
    db = MemoryDB(config.db_path)
    print(f"  Database: {config.db_path}")

    # Download the base model
    print(f"\n  Downloading base model: {config.model.name}")
    print(f"  (This may take a few minutes on first run...)")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        TRUSTED_PREFIXES = ("Qwen/", "google/gemma")
        needs_remote_code = any(config.model.name.startswith(p) for p in TRUSTED_PREFIXES)
        tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=needs_remote_code)
        model = AutoModelForCausalLM.from_pretrained(config.model.name, trust_remote_code=needs_remote_code)
        print(f"  Base model ready: {config.model.name}")
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params/1e6:.0f}M")
        del model, tokenizer  # Free memory
    except ImportError:
        print(f"  Training dependencies not installed. Run: pip install -e '.[train]'")
        print(f"  (The base model will download automatically on first training run.)")
    except Exception as e:
        print(f"  Could not download model: {e}")
        print(f"  (The base model will download automatically on first training run.)")

    print(f"\n  Initialization complete!")
    print(f"  Next steps:")
    print(f"    1. Add your OPENROUTER_API_KEY to .env")
    print(f"    2. Ingest transcripts:  dreamcatcher ingest <file>")
    print(f"    3. Run first pipeline:  dreamcatcher nightly")
    print(f"    4. Start serving:       dreamcatcher serve")
    print()


def cmd_ingest(config):
    collector = SessionCollector(config)
    if len(sys.argv) > 2:
        target = sys.argv[2]
        agent = sys.argv[3] if len(sys.argv) > 3 else "unknown"
        p = Path(target)
        if p.is_dir():
            ids = collector.ingest_directory(str(p), agent)
            print(f"Ingested {len(ids)} sessions.")
        elif p.is_file():
            sid = collector.ingest_file(str(p), agent)
            print(f"Ingested session: {sid}")
        elif target == "-":
            sid = collector.ingest_text(sys.stdin.read(), agent)
            print(f"Ingested session: {sid}")
        else:
            print(f"Not found: {target}")
            sys.exit(1)
    else:
        ids = collector.ingest_directory()
        print(f"Ingested {len(ids)} sessions from {config.sessions_dir}")


def cmd_extract(config):
    collector = SessionCollector(config)
    print("Extracting memories from unprocessed sessions...")
    memories = asyncio.run(collector.extract_memories())
    print(f"Extracted {len(memories)} memories.")


def cmd_build(config):
    builder = TrainingDataBuilder(config)
    data = builder.build_training_set()
    if not data:
        print("No training examples. Ingest and extract sessions first.")


def cmd_train(config):
    trainer = MemoryTrainer(config)
    force = "--force" in sys.argv
    result = trainer.train(force=force)
    print(json.dumps(result, indent=2))


def cmd_nightly(config):
    """Full pipeline: extract → build → train → export vault."""
    print(f"\n{'='*60}")
    print(f"  Dreamcatcher Nightly Pipeline")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # Step 1: Extract memories from new sessions
    print("Step 1/4: Extracting memories from new sessions...")
    collector = SessionCollector(config)
    memories = asyncio.run(collector.extract_memories())
    print(f"  → {len(memories)} new memories extracted\n")

    # Step 2: Build FULL training dataset (all accumulated memories)
    print("Step 2/4: Building full training dataset...")
    builder = TrainingDataBuilder(config)
    data = builder.build_training_set()
    if not data:
        print("  No personal training data yet. Skipping personal model training.")
    else:
        print()
        # Step 3: Re-fine-tune from base weights on the complete dataset
        print("Step 3/4: Re-fine-tuning memory model from base weights...")
        trainer = MemoryTrainer(config)
        result = trainer.train()

        # Step 4: Export the browsable knowledge vault
        print("\nStep 4/4: Updating knowledge vault...")
        try:
            from .wiki import WikiExporter
            exporter = WikiExporter(config)
            exporter.export()
        except Exception as e:
            print(f"  Wiki export skipped: {e}")

        print(f"\n{'='*60}")
        if result.get("status") == "success":
            print(f"  Personal pipeline complete!")
            print(f"  {result['num_examples']} examples → {result['model_name']}")
            print(f"  Loss: {result['loss_final']:.4f} | Time: {result['duration_seconds']}s")
        else:
            print(f"  Pipeline finished with status: {result.get('status')}")
        print(f"{'='*60}\n")

    # Always run team pipelines regardless of personal pipeline result
    from .teams import TeamMemoryManager
    teams_mgr = TeamMemoryManager(config)
    team_ids = teams_mgr.list_teams()
    for tid in team_ids:
        print(f"\n  Running nightly for team: {tid}")
        team_config = teams_mgr.get_config(tid)
        team_config.ensure_dirs()
        t_collector = teams_mgr.get_collector(tid)
        t_memories = asyncio.run(t_collector.extract_memories())
        print(f"    → {len(t_memories)} new memories")
        t_builder = TrainingDataBuilder(team_config)
        t_data = t_builder.build_training_set()
        if t_data:
            t_trainer = MemoryTrainer(team_config)
            t_trainer.train()


def cmd_serve(config):
    print(f"Starting Dreamcatcher server on {config.server.host}:{config.server.port}")
    from .server import run_server
    run_server()


def cmd_mcp(config):
    """Start the Dreamcatcher MCP server (stdio transport, for Claude Code)."""
    try:
        from .mcp_server import main as mcp_main
        mcp_main()
    except ImportError:
        print("Error: MCP package not installed.")
        print("Install it with: pip install dreamcatcher-memory[claude-code]")
        sys.exit(1)


def cmd_setup(config):
    """Configure Dreamcatcher integrations."""
    if len(sys.argv) < 3:
        print("Usage: dreamcatcher setup <integration>")
        print()
        print("Available integrations:")
        print("  claude-code    Configure Claude Code MCP integration")
        print("  paperclip      Create Paperclip routine for nightly training")
        print("  hermes         Create Hermes cron job for nightly training")
        print("  openclaw       Create OpenClaw cron job for nightly training")
        sys.exit(1)

    target = sys.argv[2]
    setup_map = {
        "claude-code": _setup_claude_code,
        "paperclip": _setup_paperclip,
        "hermes": _setup_hermes,
        "openclaw": _setup_openclaw,
    }
    if target in setup_map:
        setup_map[target](config)
    else:
        print(f"Unknown integration: {target}")
        print("Available: claude-code, paperclip, hermes, openclaw")
        sys.exit(1)


def _setup_claude_code(config):
    """One-command setup for the Claude Code MCP integration."""
    import shutil

    print(f"\n  Living Memory — Claude Code Setup")
    print(f"  {'─'*40}")

    # Parse flags
    args = sys.argv[3:]
    use_global = "--global" in args
    generate_claude_md = "--claude-md" in args
    server_url = "http://localhost:8420"
    for i, arg in enumerate(args):
        if arg == "--url" and i + 1 < len(args):
            server_url = args[i + 1]

    # Step 1: Health check
    print(f"\n  Checking Living Memory server at {server_url}...")
    try:
        import httpx
        resp = httpx.get(f"{server_url}/health", timeout=3.0)
        if resp.status_code == 200:
            health = resp.json()
            stats = health.get("stats", {})
            print(f"  ✓ Server reachable")
            print(f"    Model loaded: {health.get('model_loaded', False)}")
            print(f"    Active memories: {stats.get('active_memories', 0)}")
        else:
            print(f"  ⚠ Server returned {resp.status_code}")
    except Exception:
        print(f"  ⚠ Server not reachable (this is OK — configure now, start later)")

    # Step 2: Resolve the dreamcatcher command
    dc_cmd = shutil.which("dreamcatcher")
    if not dc_cmd:
        # Fallback: use the Python interpreter with -m
        # Avoid sys.executable under uv run — it points to a temp binary
        # that gets cleaned up after the command finishes.
        python_path = sys.executable
        if "/tmp/" in python_path or "/temp/" in python_path.lower():
            # Under uv run: resolve the real Python from PATH
            stable_python = shutil.which("python3") or shutil.which("python")
            if stable_python:
                python_path = stable_python
        dc_cmd = f"{python_path} -m dreamcatcher.mcp_server"
    else:
        dc_cmd = f"{dc_cmd} mcp"

    # Step 3: Build the MCP server entry
    # On macOS, the Claude Desktop app sandbox blocks files with the
    # com.apple.provenance extended attribute (set on files created by
    # "internet-downloaded" apps). Using /bin/bash -c avoids this by
    # launching from a system binary. The cd ensures config.yaml and
    # data/ paths resolve correctly.
    project_dir = str(Path(__file__).resolve().parent.parent)
    if sys.platform == "darwin":
        mcp_entry = {
            "command": "/bin/bash",
            "args": ["-c", f"cd '{project_dir}' && exec {dc_cmd}"],
            "env": {
                "DREAMCATCHER_SERVER_URL": server_url,
            },
        }
    else:
        mcp_entry = {
            "type": "stdio",
            "command": dc_cmd.split()[0],
            "args": dc_cmd.split()[1:],
            "env": {
                "DREAMCATCHER_SERVER_URL": server_url,
            },
        }

    # Step 4: Determine config paths
    # Claude Code CLI config
    if use_global:
        cli_settings_dir = Path.home() / ".claude"
    else:
        cli_settings_dir = Path.cwd() / ".claude"
    cli_settings_path = cli_settings_dir / "settings.json"

    # Claude Desktop app config (platform-specific)
    if sys.platform == "darwin":
        desktop_config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        desktop_config_path = appdata / "Claude" / "claude_desktop_config.json"
    else:
        desktop_config_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

    # Step 5: Write Claude Code CLI config
    cli_settings_dir.mkdir(parents=True, exist_ok=True)
    if cli_settings_path.exists():
        with open(cli_settings_path) as f:
            settings = json.load(f)
    else:
        settings = {}

    if "mcpServers" not in settings:
        settings["mcpServers"] = {}
    settings["mcpServers"]["Living Memory"] = mcp_entry

    with open(cli_settings_path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    scope = "global" if use_global else "project"
    print(f"\n  ✓ Claude Code (CLI) configured ({scope})")
    print(f"    {cli_settings_path}")

    # Step 6: Write Claude Desktop app config (if global and directory exists)
    if use_global and desktop_config_path.parent.exists():
        if desktop_config_path.exists():
            with open(desktop_config_path) as f:
                desktop_config = json.load(f)
        else:
            desktop_config = {}

        if "mcpServers" not in desktop_config:
            desktop_config["mcpServers"] = {}
        desktop_config["mcpServers"]["Living Memory"] = mcp_entry

        with open(desktop_config_path, "w") as f:
            json.dump(desktop_config, f, indent=2)
            f.write("\n")

        print(f"\n  ✓ Claude Desktop app configured")
        print(f"    {desktop_config_path}")

    # Step 7: Optional CLAUDE.md generation
    if generate_claude_md:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from dreamcatcher_client import generate_claude_md
            content = generate_claude_md(url=server_url)
            if content:
                print(f"  ✓ CLAUDE.md generated in current directory")
            else:
                print(f"  ⚠ Could not generate CLAUDE.md (server may not be running)")
        except Exception as e:
            print(f"  ⚠ CLAUDE.md generation failed: {e}")

    # Step 8: Success message
    print(f"\n  Setup complete! Next steps:")
    print(f"    1. Make sure the Living Memory server is running:")
    print(f"         dreamcatcher serve")
    print(f"    2. Restart Claude Code (or the Claude desktop app)")
    print(f"    3. Ask Claude to recall something about you!")
    print()
    print(f"  The MCP server will:")
    print(f"    • Inject personal memory context into every session")
    print(f"    • Provide living_memory_recall for on-demand memory queries")
    print(f"    • Auto-save conversations for nightly memory training")
    print()


def _setup_paperclip(config):
    """Create a Paperclip routine for nightly memory training."""
    args = sys.argv[3:]
    paperclip_url = "http://localhost:3000"
    company_id = None
    agent_id = None
    server_url = "http://localhost:8420"

    for i, arg in enumerate(args):
        if arg == "--url" and i + 1 < len(args):
            paperclip_url = args[i + 1]
        elif arg == "--company" and i + 1 < len(args):
            company_id = args[i + 1]
        elif arg == "--agent" and i + 1 < len(args):
            agent_id = args[i + 1]
        elif arg == "--server-url" and i + 1 < len(args):
            server_url = args[i + 1]

    if not company_id or not agent_id:
        print("Usage: dreamcatcher setup paperclip --company <id> --agent <id> [--url <paperclip-url>]")
        sys.exit(1)

    print(f"\n  Living Memory — Paperclip Setup")
    print(f"  {'─'*40}")

    try:
        import httpx
        api = paperclip_url.rstrip("/")

        # Create the routine
        print(f"\n  Creating nightly routine in Paperclip...")
        resp = httpx.post(f"{api}/api/companies/{company_id}/routines", json={
            "title": "Living Memory Nightly",
            "description": "Extract memories, build training set, and re-fine-tune the team memory model.",
            "assigneeAgentId": agent_id,
            "status": "active",
            "concurrencyPolicy": "skip_if_active",
            "catchUpPolicy": "skip_missed",
        }, timeout=10.0)
        resp.raise_for_status()
        routine = resp.json()
        routine_id = routine.get("id")
        print(f"  ✓ Routine created: {routine_id}")

        # Add schedule trigger
        print(f"  Adding 3 AM schedule trigger...")
        resp = httpx.post(f"{api}/api/routines/{routine_id}/triggers", json={
            "kind": "schedule",
            "cronExpression": "0 3 * * *",
            "timezone": "America/New_York",
            "enabled": True,
        }, timeout=10.0)
        resp.raise_for_status()
        print(f"  ✓ Schedule trigger added (3 AM daily)")

        print(f"\n  Setup complete!")
        print(f"  The routine will appear in Paperclip's Routines UI.")
        print(f"  Make sure the Living Memory server is running at {server_url}")
        print()

    except ImportError:
        print("  Error: httpx not installed. Run: pip install httpx")
        sys.exit(1)
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Make sure Paperclip is running at {paperclip_url}")
        sys.exit(1)


def _setup_hermes(config):
    """Create a Hermes cron job for nightly memory training."""
    args = sys.argv[3:]
    server_url = "http://localhost:8420"
    hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))

    for i, arg in enumerate(args):
        if arg == "--server-url" and i + 1 < len(args):
            server_url = args[i + 1]
        elif arg == "--hermes-home" and i + 1 < len(args):
            hermes_home = args[i + 1]

    print(f"\n  Living Memory — Hermes Setup")
    print(f"  {'─'*40}")

    cron_dir = Path(hermes_home) / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = cron_dir / "jobs.json"

    # Load existing jobs
    jobs = []
    if jobs_path.exists():
        with open(jobs_path) as f:
            jobs = json.load(f)

    # Remove any existing living-memory job
    jobs = [j for j in jobs if j.get("id") != "living-memory-nightly"]

    # Add the new job
    jobs.append({
        "id": "living-memory-nightly",
        "name": "Living Memory Nightly",
        "prompt": f"Run the Living Memory nightly pipeline. POST to {server_url}/nightly to trigger extraction, training, and deployment.",
        "schedule": "0 3 * * *",
        "skills": [],
        "enabled": True,
    })

    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)
        f.write("\n")

    print(f"\n  ✓ Cron job created: living-memory-nightly")
    print(f"    Schedule: 0 3 * * * (3 AM daily)")
    print(f"    Config: {jobs_path}")
    print(f"\n  View with: hermes cron list")
    print(f"  Make sure the Living Memory server is running at {server_url}")
    print()


def _setup_openclaw(config):
    """Create an OpenClaw cron job for nightly memory training."""
    import shutil
    args = sys.argv[3:]
    server_url = "http://localhost:8420"
    openclaw_home = os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw"))

    for i, arg in enumerate(args):
        if arg == "--server-url" and i + 1 < len(args):
            server_url = args[i + 1]
        elif arg == "--openclaw-home" and i + 1 < len(args):
            openclaw_home = args[i + 1]

    print(f"\n  Living Memory — OpenClaw Setup")
    print(f"  {'─'*40}")

    # Try CLI first
    openclaw_bin = shutil.which("openclaw")
    if openclaw_bin:
        import subprocess
        print(f"\n  Creating cron job via OpenClaw CLI...")
        result = subprocess.run([
            openclaw_bin, "cron", "add",
            "--name", "Living Memory Nightly",
            "--cron", "0 3 * * *",
            "--session", "isolated",
            "--message", f"Run the Living Memory nightly pipeline. POST to {server_url}/nightly to trigger extraction, training, and deployment.",
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Cron job created via CLI")
            print(f"\n  View with: openclaw cron list")
            print()
            return

    # Fallback: write directly to jobs.json
    cron_dir = Path(openclaw_home) / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = cron_dir / "jobs.json"

    jobs = []
    if jobs_path.exists():
        with open(jobs_path) as f:
            jobs = json.load(f)

    jobs = [j for j in jobs if j.get("name") != "Living Memory Nightly"]

    jobs.append({
        "name": "Living Memory Nightly",
        "message": f"Run the Living Memory nightly pipeline. POST to {server_url}/nightly to trigger extraction, training, and deployment.",
        "cron": "0 3 * * *",
        "session": "isolated",
        "enabled": True,
    })

    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)
        f.write("\n")

    print(f"\n  ✓ Cron job created: Living Memory Nightly")
    print(f"    Schedule: 0 3 * * * (3 AM daily)")
    print(f"    Config: {jobs_path}")
    print(f"\n  View with: openclaw cron list")
    print(f"  Make sure the Living Memory server is running at {server_url}")
    print()


def cmd_team(config):
    """Manage team memory pools."""
    from .teams import TeamMemoryManager

    if len(sys.argv) < 3:
        print("Usage: dreamcatcher team <list|stats|nightly> [team_id]")
        sys.exit(1)

    subcmd = sys.argv[2]
    teams = TeamMemoryManager(config)

    if subcmd == "list":
        team_ids = teams.list_teams()
        if not team_ids:
            print("  No teams found.")
        else:
            print(f"\n  Teams ({len(team_ids)}):")
            for tid in team_ids:
                s = teams.team_stats(tid)
                print(f"    {tid:30s}  {s.get('active_memories', 0)} memories, "
                      f"{s.get('unprocessed_sessions', 0)} unprocessed sessions")
            print()

    elif subcmd == "stats":
        if len(sys.argv) < 4:
            print("Usage: dreamcatcher team stats <team_id>")
            sys.exit(1)
        team_id = sys.argv[3]
        s = teams.team_stats(team_id)
        print(f"\n  Team: {team_id}")
        print(f"  {'─'*40}")
        print(f"  Sessions:          {s['total_sessions']} total, {s['unprocessed_sessions']} unprocessed")
        print(f"  Active memories:   {s['active_memories']}")
        print(f"  Training examples: {s['total_training_examples']}")
        print(f"  Training runs:     {s['training_runs']}")
        if s.get("memories_by_category"):
            print(f"\n  By category:")
            for cat, count in sorted(s["memories_by_category"].items()):
                print(f"    {cat:20s} {count}")
        print()

    elif subcmd == "nightly":
        team_id = sys.argv[3] if len(sys.argv) > 3 else None
        target_teams = [team_id] if team_id else teams.list_teams()
        if not target_teams:
            print("  No teams found.")
            return

        for tid in target_teams:
            print(f"\n{'='*60}")
            print(f"  Team Nightly Pipeline: {tid}")
            print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"{'='*60}\n")

            team_config = teams.get_config(tid)
            team_config.ensure_dirs()

            # Step 1: Extract
            print("  Step 1/3: Extracting memories...")
            collector = teams.get_collector(tid)
            memories = asyncio.run(collector.extract_memories())
            print(f"    → {len(memories)} new memories\n")

            # Step 2: Build training set
            print("  Step 2/3: Building training dataset...")
            builder = TrainingDataBuilder(team_config)
            data = builder.build_training_set()
            if not data:
                print("    No training data. Skipping training.\n")
                continue

            # Step 3: Train
            print("  Step 3/3: Re-fine-tuning team memory model...")
            trainer = MemoryTrainer(team_config)
            result = trainer.train()
            if result.get("status") == "success":
                print(f"    → {result['num_examples']} examples, loss {result['loss_final']:.4f}\n")
            else:
                print(f"    → {result.get('status')}: {result.get('reason', '?')}\n")

    else:
        print(f"Unknown team subcommand: {subcmd}")
        print("Available: list, stats, nightly")
        sys.exit(1)


def cmd_wiki(config):
    """Export the canonical memory ledger as a browsable markdown vault."""
    from .wiki import WikiExporter
    exporter = WikiExporter(config)

    if "--sync" in sys.argv:
        # Sync-only mode: apply vault edits back to SQLite without regenerating
        vault_dir = Path(config.models_dir).parent / "vault"
        print(f"\n  Syncing vault edits from {vault_dir}")
        exporter._sync_edits_from_vault(vault_dir)
        print(f"  Sync complete.\n")
    else:
        # Full export: sync edits first, then regenerate the vault
        output = None
        for i, arg in enumerate(sys.argv):
            if arg == "--output" and i + 1 < len(sys.argv):
                output = sys.argv[i + 1]

        print(f"\n  Exporting memory vault...")
        vault_path = exporter.export(output)
        print(f"  Vault ready: {vault_path}")
        print(f"  Open in Obsidian or any markdown viewer.\n")


def cmd_lint(config):
    """Run a memory consistency check across the canonical ledger."""
    from .lint import MemoryLinter
    linter = MemoryLinter(config)

    print(f"\n{'='*60}")
    print(f"  Dreamcatcher Memory Lint")
    print(f"{'='*60}\n")

    # Check if --rules-only flag is set (skip LLM pass)
    rules_only = "--rules-only" in sys.argv

    if rules_only:
        # Only run the rule-based pre-pass (zero API cost)
        memories = linter.db.get_active_memories(limit=10000)
        findings = linter._rule_based_pass(memories)
        vault = Path(config.models_dir).parent / "vault"
        vault.mkdir(parents=True, exist_ok=True)
        report_path = linter._write_report(vault, findings, len(memories))
        print(f"  Rule-based findings: {len(findings)}")
        print(f"  Report: {report_path}\n")
    else:
        # Full lint: rules + LLM
        result = linter.run_full_lint()
        print(f"\n  Total findings: {result['total']}")
        print(f"    Rule-based: {result['rule_based']}")
        print(f"    LLM-based:  {result['llm_based']}")
        print(f"  Report: {result.get('report_path', '?')}\n")


def cmd_stats(config):
    db = MemoryDB(config.db_path)
    s = db.stats()
    trainer = MemoryTrainer(config)
    model_path = trainer.get_current_model_path()

    # Show compression preview
    from .collector import TrainingDataBuilder
    comp = db.get_training_set_with_compression()

    print(f"\n  Dreamcatcher — Living Memory")
    print(f"  {'─'*40}")
    print(f"  Sessions:           {s['total_sessions']} total, {s['unprocessed_sessions']} unprocessed")
    print(f"  Active memories:    {s['active_memories']}")
    print(f"  Training examples:  {s['total_training_examples']} total in database")
    print(f"    Recent (<6mo):    {s.get('recent_examples', '?')} (full episodic density)")
    print(f"    Old (>6mo):       {s.get('old_examples', '?')} ({comp['n_compressed']} kept, {comp['n_dropped']} compressed out)")
    print(f"    Nightly set size: {len(comp['examples'])} ({len(comp['examples'])/max(s['total_training_examples'],1)*100:.0f}% of total)")
    print(f"  Training runs:      {s['training_runs']}")
    print(f"  Current model:      {model_path or '(none — run training first)'}")
    if s.get("memories_by_category"):
        print(f"\n  By category:")
        for cat, count in sorted(s["memories_by_category"].items()):
            print(f"    {cat:20s} {count}")
    print()


def cmd_export(config):
    db = MemoryDB(config.db_path)
    memories = db.get_active_memories(limit=10000)
    examples = db.get_all_training_examples()
    output = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "memories": memories,
        "training_examples": examples,
    }
    out_path = Path("data") / "export.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Exported {len(memories)} memories + {len(examples)} training examples → {out_path}")


def cmd_cleanup(config):
    """Remove old model checkpoints, keeping the N most recent."""
    keep = 3
    for i, arg in enumerate(sys.argv):
        if arg == "--keep" and i + 1 < len(sys.argv):
            keep = int(sys.argv[i + 1])

    models_dir = Path(config.models_dir)
    checkpoints = sorted(
        [d for d in models_dir.glob("memory_*") if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )

    if len(checkpoints) <= keep:
        print(f"Only {len(checkpoints)} checkpoints. Nothing to clean up.")
        return

    to_remove = checkpoints[keep:]
    print(f"Removing {len(to_remove)} old checkpoints (keeping {keep}):")
    import shutil
    for d in to_remove:
        print(f"  rm {d.name}")
        shutil.rmtree(d)
    print("Done.")


def cmd_update(config):
    """Pull latest code and reinstall dependencies."""
    import subprocess
    import shutil

    project_dir = Path(__file__).resolve().parent.parent
    is_source_checkout = (project_dir / ".git").exists()
    is_editable = (project_dir / "pyproject.toml").exists() and is_source_checkout

    print(f"\n  Living Memory — Update")
    print(f"  {'─'*40}")

    if is_editable:
        # Source checkout: git pull + pip install -e .
        print(f"\n  Detected source checkout at {project_dir}")
        print(f"  Pulling latest code...")
        result = subprocess.run(
            ["git", "pull"], cwd=str(project_dir),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
        else:
            print(f"  Git pull failed: {result.stderr.strip()}")
            print(f"  Continuing with reinstall...")

        print(f"\n  Reinstalling from source...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=str(project_dir), capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  Package reinstalled.")
        else:
            print(f"  Install failed: {result.stderr.strip()}")
            return
    else:
        # Installed via pip/wheel: upgrade from PyPI
        print(f"\n  Detected pip-installed package. Upgrading...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "dreamcatcher-memory"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  Package upgraded.")
        else:
            print(f"  Upgrade failed: {result.stderr.strip()}")
            return

    # Show new version
    try:
        from importlib import reload
        import dreamcatcher as dc
        reload(dc)
        print(f"\n  Updated to version {dc.__version__}")
    except Exception:
        pass

    print(f"\n  Update complete!")
    print(f"  Restart 'dreamcatcher serve' if it's running.\n")


def cmd_uninstall(config):
    """Remove Living Memory configs, integrations, and optionally data."""
    import shutil

    print(f"\n  Living Memory — Uninstall")
    print(f"  {'─'*40}")

    removed = []

    # Step 1: Remove Claude Code MCP config
    for settings_path in [
        Path.home() / ".claude" / "settings.json",
        Path.cwd() / ".claude" / "settings.json",
    ]:
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    settings = json.load(f)
                if "mcpServers" in settings and "Living Memory" in settings["mcpServers"]:
                    del settings["mcpServers"]["Living Memory"]
                    with open(settings_path, "w") as f:
                        json.dump(settings, f, indent=2)
                        f.write("\n")
                    removed.append(f"Claude Code MCP config ({settings_path})")
            except Exception:
                pass

    # Step 2: Remove Claude Desktop config
    if sys.platform == "darwin":
        desktop_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        desktop_path = appdata / "Claude" / "claude_desktop_config.json"
    else:
        desktop_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

    if desktop_path.exists():
        try:
            with open(desktop_path) as f:
                dc_config = json.load(f)
            if "mcpServers" in dc_config and "Living Memory" in dc_config["mcpServers"]:
                del dc_config["mcpServers"]["Living Memory"]
                with open(desktop_path, "w") as f:
                    json.dump(dc_config, f, indent=2)
                    f.write("\n")
                removed.append(f"Claude Desktop config ({desktop_path})")
        except Exception:
            pass

    # Step 3: Remove Claude Code scheduled task
    sched_dir = Path.home() / ".claude" / "scheduled-tasks" / "living-memory-nightly"
    if sched_dir.exists():
        shutil.rmtree(sched_dir)
        removed.append("Claude Code scheduled task")

    # Step 4: Remove Hermes cron job
    hermes_jobs = Path.home() / ".hermes" / "cron" / "jobs.json"
    if hermes_jobs.exists():
        try:
            with open(hermes_jobs) as f:
                jobs = json.load(f)
            before = len(jobs)
            jobs = [j for j in jobs if j.get("id") != "living-memory-nightly"]
            if len(jobs) < before:
                with open(hermes_jobs, "w") as f:
                    json.dump(jobs, f, indent=2)
                    f.write("\n")
                removed.append("Hermes cron job")
        except Exception:
            pass

    # Step 5: Remove OpenClaw cron job
    openclaw_jobs = Path.home() / ".openclaw" / "cron" / "jobs.json"
    if openclaw_jobs.exists():
        try:
            with open(openclaw_jobs) as f:
                jobs = json.load(f)
            before = len(jobs)
            jobs = [j for j in jobs if j.get("name") != "Living Memory Nightly"]
            if len(jobs) < before:
                with open(openclaw_jobs, "w") as f:
                    json.dump(jobs, f, indent=2)
                    f.write("\n")
                removed.append("OpenClaw cron job")
        except Exception:
            pass

    # Step 6: Warn about Paperclip (remote routine can't be removed locally)
    # Check if Paperclip was ever configured by looking for the plugin
    paperclip_plugin = Path(__file__).parent.parent / "integrations" / "paperclip"
    if paperclip_plugin.exists():
        print(f"\n  Note: If you configured a Paperclip routine, it must be")
        print(f"  removed from the Paperclip UI or API separately.")
        print(f"  Living Memory cannot delete remote Paperclip routines.")

    if removed:
        print(f"\n  Removed integrations:")
        for r in removed:
            print(f"    - {r}")
    else:
        print(f"\n  No local integration configs found to remove.")

    # Step 6: Optionally remove data
    data_dir = Path(config.db_path).parent
    if data_dir.exists():
        print(f"\n  Data directory: {data_dir}")
        db = MemoryDB(config.db_path)
        stats = db.stats()
        print(f"    {stats.get('active_memories', 0)} memories, "
              f"{stats.get('total_sessions', 0)} sessions, "
              f"{stats.get('training_runs', 0)} training runs")

        if _confirm("Delete all memory data? This cannot be undone", default=False):
            shutil.rmtree(data_dir)
            print(f"  Data deleted.")
        else:
            print(f"  Data preserved at {data_dir}")

    print(f"\n  Uninstall complete.")
    print(f"  To remove the Python package: pip uninstall dreamcatcher-memory\n")


def cmd_reinstall(config):
    """Full uninstall + quickstart in one command."""
    print(f"\n  Living Memory — Reinstall")
    print(f"  {'─'*40}")
    print(f"  This will remove all configs and re-run the setup wizard.\n")

    if not _confirm("Proceed with reinstall?"):
        print(f"  Cancelled.\n")
        return

    cmd_uninstall(config)
    print()
    cmd_quickstart(config)


if __name__ == "__main__":
    main()
