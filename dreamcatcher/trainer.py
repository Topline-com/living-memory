"""
Dreamcatcher Trainer — Dual Backend (PyTorch + MLX)
====================================================
Nightly full re-fine-tuning of a compact model from fixed pretrained
base weights on the complete rendered training set.

The trainer auto-detects the platform and uses the appropriate backend:
  - Apple Silicon (M1/M2/M3/M4): MLX via mlx-lm (recommended for Mac)
  - NVIDIA GPU: PyTorch via HuggingFace Transformers
  - CPU fallback: PyTorch (slow but works)

Each training run starts from the ORIGINAL pretrained weights (M_0),
never from a previous checkpoint. This is the core invariant:
M_t = F(M_0, T_t) — no sequential-update drift.
"""
import json
import time
import shutil
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .database import MemoryDB
from .config import DreamcatcherConfig


def _detect_backend() -> str:
    """Auto-detect the best training backend for this machine."""
    # Check for Apple Silicon first
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core
            return "mlx"
        except ImportError:
            pass

    # Check for NVIDIA GPU
    try:
        import torch
        if torch.cuda.is_available():
            return "pytorch-cuda"
    except ImportError:
        pass

    # CPU fallback (slow but works)
    try:
        import torch
        return "pytorch-cpu"
    except ImportError:
        pass

    return "none"


class MemoryTrainer:
    """
    Full re-fine-tuning of a compact model on the complete memory dataset.
    Called nightly by cron. No incremental updates, no adapters.
    Each run produces a fresh model from the original pretrained base.
    """

    def __init__(self, config: DreamcatcherConfig = None):
        self.config = config or DreamcatcherConfig.load()
        self.db = MemoryDB(self.config.db_path)
        self.backend = _detect_backend()

    def train(self, training_data_path: Optional[str] = None, force: bool = False) -> dict:
        """
        Re-fine-tune the memory model from pretrained base weights.

        The workflow:
        1. Load the ORIGINAL pretrained base weights (M_0) — never a checkpoint
        2. Fine-tune on the complete rendered training set (T_t)
        3. Save the trained model — this IS the memory
        4. Atomic symlink swap to make it the active model
        """
        start_time = time.time()
        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        # ── Find training data ──────────────────────────────────
        if training_data_path:
            data_path = Path(training_data_path)
        else:
            data_path = Path(self.config.training_dir) / f"full_dataset_{today}.jsonl"

        if not data_path.exists():
            print(f"  No training data at {data_path}")
            return {"status": "skipped", "reason": "no_data"}

        with open(data_path) as f:
            examples = [json.loads(line) for line in f if line.strip()]

        if len(examples) < 5 and not force:
            print(f"  Only {len(examples)} examples. Waiting for more data (use --force to override).")
            return {"status": "skipped", "reason": "too_few_examples", "count": len(examples)}

        print(f"\n{'='*60}")
        print(f"  Dreamcatcher Training — {today}")
        print(f"  Re-fine-tuning on {len(examples)} examples")
        print(f"  Backend: {self.backend}")
        print(f"{'='*60}\n")

        # ── Route to the appropriate backend ─────────────────────
        if self.backend == "mlx":
            result = self._train_mlx(examples, today)
        elif self.backend in ("pytorch-cuda", "pytorch-cpu"):
            result = self._train_pytorch(examples, today)
        else:
            print("  ERROR: No training backend available.")
            print("  Install either: pip install mlx mlx-lm (Apple Silicon)")
            print("                  pip install torch transformers (NVIDIA/CPU)")
            return {"status": "error", "reason": "no_backend"}

        # ── Log the run ─────────────────────────────────────────
        if result.get("status") == "success":
            duration = time.time() - start_time
            result["duration_seconds"] = round(duration, 1)
            self.db.log_training_run(
                model_path=result.get("model_path", ""),
                num_examples=len(examples),
                loss_final=result.get("loss_final", 0),
                duration_seconds=round(duration, 1),
                model_name=self.config.model.name,
            )
            print(f"\n  Training complete!")
            print(f"  {len(examples)} examples → {result.get('model_path', '?')}")
            print(f"  Loss: {result.get('loss_final', 0):.4f} | Time: {duration:.1f}s")

        return result

    # ══════════════════════════════════════════════════════════════
    # MLX Backend (Apple Silicon)
    # ══════════════════════════════════════════════════════════════

    def _train_mlx(self, examples: list, today: str) -> dict:
        """
        Full re-fine-tuning using MLX on Apple Silicon.
        Uses mlx-lm's training utilities with the full model (not LoRA).
        Apple Silicon's unified memory makes this feasible for sub-3B models.
        """
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        try:
            from mlx_lm import load as mlx_load
            from mlx_lm.tuner import train as mlx_train
            from mlx_lm.tuner.trainer import TrainingArgs
        except ImportError:
            print("  ERROR: mlx-lm not installed. Run: pip install mlx-lm")
            return {"status": "error", "reason": "mlx_lm_not_installed"}

        model_name = self.config.model.name

        # ── Prepare training data as JSONL ───────────────────────
        # mlx-lm expects a directory with train.jsonl
        train_dir = Path(self.config.training_dir) / f"mlx_{today}"
        train_dir.mkdir(parents=True, exist_ok=True)
        train_file = train_dir / "train.jsonl"

        with open(train_file, "w") as f:
            for ex in examples:
                # mlx-lm expects {"text": "..."} format
                # We format messages into a single text string
                msgs = ex.get("messages", [])
                text_parts = []
                for msg in msgs:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text_parts.append(f"<|{role}|>\n{content}")
                text_parts.append("<|assistant|>")
                f.write(json.dumps({"text": "\n".join(text_parts)}) + "\n")

        print(f"  Loading base model: {model_name} (MLX)")
        print(f"  Training data: {len(examples)} examples → {train_file}")

        # ── Load model and tokenizer ──────────────────────────────
        model, tokenizer = mlx_load(model_name)

        # ── Configure training ────────────────────────────────────
        # Full fine-tuning: all parameters trainable (no LoRA adapters)
        tc = self.config.training
        training_args = TrainingArgs(
            batch_size=tc.batch_size,
            iters=len(examples) * tc.epochs // tc.batch_size,
            val_batches=0,  # No validation split — we benchmark separately
            learning_rate=tc.learning_rate,
            steps_per_report=10,
            steps_per_eval=0,
            save_every=999999,  # Don't save intermediate checkpoints
            max_seq_length=self.config.model.max_seq_length,
            grad_checkpoint=True,  # Critical for fitting 2.3B in 24GB
        )

        # ── Output directory ──────────────────────────────────────
        output_dir = Path(self.config.models_dir) / f"training_mlx_{today}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Training with MLX (full fine-tune, {tc.epochs} epochs)...")

        # ── Run training ──────────────────────────────────────────
        try:
            mlx_train(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_file,
            )
            loss_final = 0.0  # mlx_train doesn't return loss directly; we log from console
        except Exception as e:
            print(f"  MLX training error: {e}")
            # Fallback: try using mlx-lm CLI approach
            return self._train_mlx_cli(examples, today, train_file)

        # ── Save the trained model ────────────────────────────────
        final_dir = Path(self.config.models_dir) / f"memory_{today}"
        print(f"  Saving trained model to {final_dir}")

        # Save MLX model weights and tokenizer
        model.save_weights(str(final_dir / "weights.safetensors"))
        tokenizer.save_pretrained(str(final_dir))
        # Copy config for loading
        import shutil as sh
        for cfg_file in (output_dir / "config.json", Path(model_name) / "config.json"):
            if cfg_file.exists():
                sh.copy2(str(cfg_file), str(final_dir / "config.json"))
                break

        # ── Atomic symlink swap ───────────────────────────────────
        self._swap_model(final_dir)

        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        if train_dir.exists():
            shutil.rmtree(train_dir, ignore_errors=True)

        return {
            "status": "success",
            "model_path": str(final_dir),
            "num_examples": len(examples),
            "loss_final": loss_final,
            "model_name": self.config.model.name,
            "backend": "mlx",
        }

    def _train_mlx_cli(self, examples: list, today: str, train_file: Path) -> dict:
        """
        Fallback: use mlx-lm's CLI fine-tuning if the Python API fails.
        This shells out to `mlx_lm.lora` which handles memory management.
        """
        import subprocess

        final_dir = Path(self.config.models_dir) / f"memory_{today}"
        final_dir.mkdir(parents=True, exist_ok=True)
        tc = self.config.training
        iters = len(examples) * tc.epochs // max(tc.batch_size, 1)

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self.config.model.name,
            "--train",
            "--data", str(train_file.parent),
            "--batch-size", str(tc.batch_size),
            "--iters", str(iters),
            "--learning-rate", str(tc.learning_rate),
            "--adapter-path", str(final_dir),
            "--fine-tune-type", "full",  # Full re-fine-tuning, not LoRA
            "--grad-checkpoint",          # Critical for fitting ~2.3B in 24GB unified memory
            "--num-layers", "-1",         # All layers (explicit full fine-tune)
            "--seed", "42",               # Reproducibility
        ]

        print(f"  Falling back to mlx_lm CLI: {' '.join(cmd[-6:])}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  CLI training failed: {result.stderr[:500]}")
            return {"status": "error", "reason": "mlx_cli_failed"}

        self._swap_model(final_dir)
        return {
            "status": "success",
            "model_path": str(final_dir),
            "num_examples": len(examples),
            "loss_final": 0.0,
            "model_name": self.config.model.name,
            "backend": "mlx-cli",
        }

    # ══════════════════════════════════════════════════════════════
    # PyTorch Backend (NVIDIA CUDA / CPU)
    # ══════════════════════════════════════════════════════════════

    def _train_pytorch(self, examples: list, today: str) -> dict:
        """
        Full re-fine-tuning using PyTorch + HuggingFace Transformers.
        Works on NVIDIA GPUs (fast) and CPU (slow but functional).
        """
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        )
        from datasets import Dataset

        # ── Load the BASE model (always from pretrained, never a checkpoint) ──
        model, tokenizer = self._load_base_pytorch()
        if model is None:
            return {"status": "error", "reason": "model_load_failed"}

        # ── Prepare dataset ───────────────────────────────────────
        def format_for_training(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )
            return {"text": text}

        dataset = Dataset.from_list(examples)
        dataset = dataset.map(format_for_training)

        print(f"  Dataset: {len(dataset)} examples")
        print(f"  Model: {self.config.model.name}")

        # ── Train ─────────────────────────────────────────────────
        output_dir = Path(self.config.models_dir) / f"training_{today}"
        output_dir.mkdir(parents=True, exist_ok=True)
        tc = self.config.training

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=tc.epochs,
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            warmup_ratio=tc.warmup_ratio,
            weight_decay=tc.weight_decay,
            max_grad_norm=tc.max_grad_norm,
            logging_steps=5,
            save_strategy="no",
            fp16=tc.fp16 and torch.cuda.is_available(),
            remove_unused_columns=False,
            dataloader_pin_memory=torch.cuda.is_available(),
            seed=42,
        )

        def tokenize_fn(example):
            return tokenizer(
                example["text"], truncation=True,
                max_length=self.config.model.max_seq_length, padding=False,
            )

        tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        print("  Training...")
        train_result = trainer.train()
        loss_final = train_result.training_loss

        # ── Save and swap ─────────────────────────────────────────
        final_dir = Path(self.config.models_dir) / f"memory_{today}"
        print(f"  Saving trained model to {final_dir}")
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        self._swap_model(final_dir)

        if output_dir.exists() and output_dir != final_dir:
            shutil.rmtree(output_dir, ignore_errors=True)

        return {
            "status": "success",
            "model_path": str(final_dir),
            "num_examples": len(examples),
            "loss_final": round(loss_final, 4),
            "model_name": self.config.model.name,
            "backend": self.backend,
        }

    def _load_base_pytorch(self):
        """Load the pretrained base model via PyTorch. Always from original weights."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        models_to_try = [self.config.model.name] + (self.config.model.fallbacks or [])

        for name in models_to_try:
            try:
                print(f"  Loading base model: {name}")
                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32,
                    trust_remote_code=True,
                    device_map="auto",
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    model.config.pad_token_id = tokenizer.eos_token_id

                param_count = sum(p.numel() for p in model.parameters())
                print(f"  Loaded {name} ({param_count/1e6:.0f}M parameters)")
                return model, tokenizer
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
                continue

        print("  ERROR: Could not load any model.")
        return None, None

    # ══════════════════════════════════════════════════════════════
    # Common utilities
    # ══════════════════════════════════════════════════════════════

    def _swap_model(self, final_dir: Path):
        """Atomic symlink swap: update 'current' to point to the new model."""
        current_link = Path(self.config.models_dir) / "current"

        if current_link.is_symlink() or current_link.exists():
            old_target = current_link.resolve() if current_link.is_symlink() else current_link
            backup = Path(self.config.models_dir) / "previous"
            if backup.exists():
                shutil.rmtree(backup)
            if old_target.exists() and old_target != final_dir:
                shutil.move(str(old_target), str(backup))
            current_link.unlink(missing_ok=True)

        current_link.symlink_to(final_dir.resolve())
        print(f"  Updated current → {final_dir.name}")

    def get_current_model_path(self) -> Optional[str]:
        """Get the path to the currently trained model."""
        current = Path(self.config.models_dir) / "current"
        if current.exists():
            resolved = current.resolve()
            if resolved.exists():
                return str(resolved)
        return None
