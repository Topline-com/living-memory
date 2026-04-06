"""
Dreamcatcher — Living Memory for AI Agents
========================================================
Living memory: memory that lives in the model's weights, not in a
database. Memory that grows and reconsolidates every night while you
sleep. The Dreamcatcher architecture makes this possible by re-fine-
tuning a compact model from fixed pretrained weights every night on
the complete canonical memory ledger.

Default base model: Gemma 4 E2B (Google, Apache 2.0)
  - Native structured JSON output
  - ~2.3B effective parameters, PLE tables frozen during fine-tuning
  - MLX on Apple Silicon, PyTorch on NVIDIA

Architecture:
  Sessions → Memory Extraction (frontier LLM) → SQLite
  → Full Training Dataset → Nightly Re-Fine-Tuning from Base Weights
  → Inference Server → Agent Integration
  → Browsable Knowledge Vault (Obsidian-compatible)
"""
__version__ = "0.4.0"

from .config import DreamcatcherConfig
from .database import MemoryDB
from .collector import SessionCollector, TrainingDataBuilder
from .trainer import MemoryTrainer
from .wiki import WikiExporter
from .lint import MemoryLinter

__all__ = [
    "DreamcatcherConfig",
    "MemoryDB",
    "SessionCollector",
    "TrainingDataBuilder",
    "MemoryTrainer",
    "WikiExporter",
    "MemoryLinter",
]
