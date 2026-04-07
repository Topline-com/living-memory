"""Dreamcatcher Configuration — Living Memory for AI Agents."""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Gemma 4 E2B: ~2.3B effective parameters, strong structured JSON output,
    # function calling, agentic workflows. PLE tables frozen during fine-tuning.
    name: str = "google/gemma-4-E2B-it"
    fallbacks: list = field(default_factory=lambda: [
        "Qwen/Qwen3.5-0.8B",                      # 800M dense, guaranteed compatibility
        "HuggingFaceTB/SmolLM2-360M-Instruct",     # Ultra-light: CPU-feasible
    ])
    max_seq_length: int = 2048
    output_format: str = "json"


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6  # 5e-6 for Gemma 4 E2B; increase to 2e-5 for Qwen3.5-0.8B
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True


@dataclass
class ExtractionConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8420
    stale_model_threshold_hours: float = 36.0


@dataclass
class DreamcatcherConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    db_path: str = "./data/memory.db"
    sessions_dir: str = "./data/sessions"
    training_dir: str = "./data/training"
    models_dir: str = "./data/models"

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "DreamcatcherConfig":
        # Resolve config path relative to this file so the CLI works from any cwd
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            # First try relative to cwd (explicit path), then fall back to package dir
            if not config_path_obj.exists():
                config_path_obj = Path(__file__).parent.parent / config_path
        path = config_path_obj
        raw = yaml.safe_load(open(path)) if path.exists() else {}
        # Anchor for resolving relative data paths: directory containing config.yaml
        base_dir = path.parent.resolve() if path.exists() else Path(__file__).parent.parent.resolve()
        cfg = cls()
        # Resolve default relative paths against base_dir so they work from any cwd
        for attr in ("db_path", "sessions_dir", "training_dir", "models_dir"):
            val = getattr(cfg, attr)
            if not Path(val).is_absolute():
                setattr(cfg, attr, str((base_dir / val).resolve()))
        if "model" in raw:
            cfg.model = ModelConfig(**{k: v for k, v in raw["model"].items()
                                       if k in ModelConfig.__dataclass_fields__})
        if "training" in raw:
            cfg.training = TrainingConfig(**{k: v for k, v in raw["training"].items()
                                             if k in TrainingConfig.__dataclass_fields__})
        if "extraction" in raw:
            cfg.extraction = ExtractionConfig(**{k: v for k, v in raw["extraction"].items()
                                                  if k in ExtractionConfig.__dataclass_fields__})
        if "server" in raw:
            cfg.server = ServerConfig(**{k: v for k, v in raw["server"].items()
                                         if k in ServerConfig.__dataclass_fields__})
        if "paths" in raw:
            for attr in ("db_path", "sessions_dir", "training_dir", "models_dir"):
                if attr in raw["paths"]:
                    val = raw["paths"][attr]
                    if not Path(val).is_absolute():
                        val = str((base_dir / val).resolve())
                    setattr(cfg, attr, val)
        return cfg

    def ensure_dirs(self):
        """Create data directories if they don't exist."""
        for d in [self.sessions_dir, self.training_dir, self.models_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
