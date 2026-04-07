"""
Dreamcatcher Inference Server
==========================
Loads the trained memory model and serves queries over HTTP.
Agents call this to get personalized context as structured JSON.

The memory model runs on-device. During inference, personal knowledge
never leaves the device — only curated structured context is returned
for injection into a frontier reasoning model's prompt.

Includes disaster recovery: if the deployed model is stale (nightly
pipeline failed), recent untrained memories are injected as structured
context alongside the model's parametric output.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from .config import DreamcatcherConfig
from .database import MemoryDB
from .collector import SessionCollector

# ── Global state ────────────────────────────────────────────────────
_model = None
_tokenizer = None
_backend = None  # "mlx" or "pytorch"
_config = None
_db = None
_collector = None


def _load_model(config: DreamcatcherConfig):
    """Load the trained memory model (not the base — the fine-tuned one)."""
    global _model, _tokenizer, _backend

    current_model = Path(config.models_dir) / "current"
    if not current_model.exists():
        print("  No trained model found. Run 'python -m dreamcatcher train' first.")
        print("  Server will run in database-only mode.")
        return

    model_path = current_model.resolve()
    print(f"  Loading trained memory model from {model_path}")

    # ── Detect MLX adapter format (produced by mlx_lm.lora --fine-tune-type full) ──
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        try:
            with open(adapter_config) as f:
                acfg = json.load(f)
            base_model = acfg.get("model", config.model.name)
            print(f"  Detected MLX adapter format (base: {base_model})")

            from mlx_lm import load as mlx_load

            _model, _tokenizer = mlx_load(
                base_model,
                adapter_path=str(model_path),
            )
            _backend = "mlx"
            print(f"  Model loaded via MLX, ready for inference")
            return
        except Exception as e:
            print(f"  MLX loading failed: {e}")
            print("  Trying PyTorch fallback...")

    # ── PyTorch loading (for models saved as full HF checkpoints) ──
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        _model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _model.eval()

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _backend = "pytorch"
        param_count = sum(p.numel() for p in _model.parameters())
        print(f"  Model loaded via PyTorch: {param_count/1e6:.0f}M params, ready for inference")

    except Exception as e:
        print(f"  ERROR loading model: {e}")
        print("  Server will run in database-only mode.")
        _model = None
        _tokenizer = None
        _backend = None


def create_app(config: DreamcatcherConfig = None) -> "FastAPI":
    from fastapi import FastAPI
    from pydantic import BaseModel

    config = config or DreamcatcherConfig.load()

    @asynccontextmanager
    async def lifespan(app):
        global _config, _db, _collector
        _config = config
        _db = MemoryDB(config.db_path)
        _collector = SessionCollector(config)
        _load_model(config)
        yield
        global _model, _tokenizer
        _model = None
        _tokenizer = None

    app = FastAPI(
        title="Dreamcatcher — Personal Memory LLM",
        description="Query your personal memory model",
        version="0.2.0",
        lifespan=lifespan,
    )

    # ── Request/Response Models ─────────────────────────────────

    class RecallRequest(BaseModel):
        query: str
        max_tokens: int = 256

    class ContextRequest(BaseModel):
        query: str
        agent_name: str = "unknown"
        max_tokens: int = 512

    class IngestRequest(BaseModel):
        transcript: str
        agent_name: str = "unknown"
        extract_now: bool = False

    class MemoryResponse(BaseModel):
        response: str
        memories: list = []
        source: str = "none"
        latency_ms: float = 0

    # ── Endpoints ───────────────────────────────────────────────

    @app.get("/health")
    async def health():
        model_age_hours = None
        model_path = Path(config.models_dir) / "current"
        if model_path.exists():
            try:
                model_date = datetime.fromtimestamp(
                    model_path.resolve().stat().st_mtime, tz=timezone.utc
                )
                model_age_hours = round(
                    (datetime.now(timezone.utc) - model_date).total_seconds() / 3600, 1
                )
            except Exception:
                pass

        return {
            "status": "ok",
            "model_loaded": _model is not None,
            "model_backend": _backend,
            "model_path": str(model_path),
            "model_age_hours": model_age_hours,
            "stats": _db.stats() if _db else {},
        }

    @app.post("/recall", response_model=MemoryResponse)
    async def recall(req: RecallRequest):
        """
        Ask the memory model a question about the user.
        Returns structured memories as JSON.
        """
        start = time.time()

        # Try the trained model first
        memories = []
        source = "none"

        if _model and _tokenizer:
            raw = _generate(req.query, req.max_tokens)
            memories = _parse_memories(raw)
            source = "model"

        # Fall back to / supplement with database search
        if not memories and _db:
            db_memories = _search_db(req.query)
            memories = db_memories
            source = "database" if not source == "model" else "hybrid"
        elif _db:
            db_memories = _search_db(req.query)
            if db_memories:
                # Merge, dedup by content similarity
                seen = {m.get("content", "")[:50] for m in memories}
                for m in db_memories:
                    if m.get("content", "")[:50] not in seen:
                        memories.append(m)
                source = "hybrid"

        latency = (time.time() - start) * 1000
        # Build a human-readable response string
        response_text = "\n".join(
            f"[{m.get('category', '?')}] {m.get('content', '')}"
            for m in memories
        ) if memories else "No memories found for this query."

        return MemoryResponse(
            response=response_text,
            memories=memories,
            source=source,
            latency_ms=round(latency, 1),
        )

    @app.post("/context", response_model=MemoryResponse)
    async def get_context(req: ContextRequest):
        """
        Get a structured context block to inject into an agent's prompt.
        This is the primary integration endpoint.

        Returns formatted text ready to insert into a system prompt,
        plus the raw structured memories for agents that want them.
        """
        start = time.time()
        all_memories = []
        source = "none"

        # Gather memories from the model
        if _model and _tokenizer:
            # Ask several angles to get comprehensive context
            queries = [
                req.query,
                "What are the user's active projects and priorities?",
                "What are the user's communication and working style preferences?",
            ]
            for q in queries:
                raw = _generate(q, max_tokens=req.max_tokens // len(queries))
                all_memories.extend(_parse_memories(raw))
            source = "model"

        # Also pull query-relevant memories from database
        if _db:
            db_mems = _search_db(req.query, limit=20)
            if db_mems:
                all_memories.extend(db_mems)
                if source == "model":
                    source = "hybrid"
                else:
                    source = "database"

        # ── Disaster Recovery: Delta Context Injection ─────────────
        # If the deployed model is stale (>24h since last training),
        # inject recent core_facts that the model hasn't been trained on.
        # This is STRICTLY quarantined as disaster recovery — it only
        # activates when the nightly pipeline has failed.
        delta_facts = []
        if _model and _db:
            model_path = Path(config.models_dir) / "current"
            if model_path.exists():
                model_date = datetime.fromtimestamp(
                    model_path.resolve().stat().st_mtime, tz=timezone.utc
                )
                hours_stale = (datetime.now(timezone.utc) - model_date).total_seconds() / 3600
                if hours_stale > 36:  # Model is more than 36h old — pipeline likely failed
                    # Pull memories created after the model was trained
                    from .database import MemoryDB
                    with _db._conn() as conn:
                        recent = conn.execute(
                            "SELECT category, content FROM memories WHERE active = 1 AND created_at > ? ORDER BY created_at DESC LIMIT 30",
                            (model_date.isoformat(),)
                        ).fetchall()
                    delta_facts = [dict(r) for r in recent]
                    if delta_facts:
                        source = "model+delta_recovery"

        # Deduplicate by content prefix
        seen = set()
        unique = []
        for m in all_memories:
            key = m.get("content", "")[:60]
            if key not in seen:
                seen.add(key)
                unique.append(m)
        all_memories = unique

        # Format as injectable context block
        if all_memories or delta_facts:
            by_cat = {}
            for m in all_memories:
                cat = m.get("category", "other")
                if cat not in by_cat:
                    by_cat[cat] = []
                by_cat[cat].append(m.get("content", ""))

            lines = ["<personal_memory>"]
            for cat in ("fact", "project", "preference", "pattern", "relationship", "decision", "other"):
                if cat in by_cat:
                    lines.append(f"[{cat.upper()}]")
                    for item in by_cat[cat][:10]:
                        lines.append(f"  {item}")

            # Delta injection: recent facts the model hasn't been trained on
            if delta_facts:
                lines.append("[RECENT — not yet consolidated into memory model]")
                for df in delta_facts:
                    lines.append(f"  [{df['category'].upper()}] {df['content']}")

            lines.append("</personal_memory>")
            response_text = "\n".join(lines)
        else:
            response_text = ""

        latency = (time.time() - start) * 1000
        return MemoryResponse(
            response=response_text,
            memories=all_memories,
            source=source,
            latency_ms=round(latency, 1),
        )

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        session_id = _collector.ingest_text(req.transcript, req.agent_name)
        result = {"session_id": session_id, "status": "stored"}
        if req.extract_now:
            try:
                memories = await _collector.extract_memories(session_id)
                result["status"] = "extracted"
                result["memories_extracted"] = len(memories)
            except Exception as e:
                result["extraction_error"] = str(e)
        return result

    @app.get("/memories")
    async def list_memories(category: Optional[str] = None, limit: int = 50):
        memories = _db.get_active_memories(category=category, limit=limit)
        return {"memories": memories, "count": len(memories)}

    @app.get("/stats")
    async def stats():
        return _db.stats() if _db else {}

    return app


# ── Model inference helpers ─────────────────────────────────────────

SYSTEM_MSG = (
    "You are a personal memory retrieval system. Given a question about the user, "
    "respond with a JSON object: {\"memories\": [{\"category\": \"...\", \"content\": \"...\", "
    "\"confidence\": 0.0-1.0}]}. Be precise and factual."
)


def _generate(query: str, max_tokens: int = 256) -> str:
    """Generate a response from the trained memory model."""
    if not _model or not _tokenizer:
        return ""

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": query},
    ]

    if _backend == "mlx":
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        if hasattr(_tokenizer, "apply_chat_template"):
            prompt = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{query}\n<|assistant|>\n"

        sampler = make_sampler(temp=0.3, top_p=0.9)
        return mlx_generate(
            _model, _tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    # PyTorch path
    input_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(input_text, return_tensors="pt").to(_model.device)

    import torch
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()


def _parse_memories(raw: str) -> list[dict]:
    """Parse the model's JSON output into a list of memory dicts."""
    if not raw:
        return []
    try:
        # Try parsing as JSON directly
        data = json.loads(raw)
        if isinstance(data, dict) and "memories" in data:
            return data["memories"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # Try extracting JSON from the response
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            if isinstance(data, dict) and "memories" in data:
                return data["memories"]
        except (ValueError, json.JSONDecodeError):
            pass
    return []


def _search_db(query: str, limit: int = 10) -> list[dict]:
    """Simple keyword search over the memory database.

    TODO: Upgrade to embedding-based search using all-MiniLM-L6-v2
    (already configured in config.yaml for dedup). Word-overlap is a
    stopgap that misses semantic similarity. See issue #XX.
    """
    if not _db:
        return []
    all_memories = _db.get_active_memories(limit=200)
    query_words = set(query.lower().split())
    scored = []
    for m in all_memories:
        content_lower = m["content"].lower()
        hits = sum(1 for w in query_words if w in content_lower)
        if hits > 0:
            scored.append((hits, m))
    scored.sort(key=lambda x: -x[0])
    return [
        {"category": m["category"], "content": m["content"],
         "confidence": m.get("confidence", 1.0)}
        for _, m in scored[:limit]
    ]


# ── Standalone runner ───────────────────────────────────────────────

def run_server(config_path: str = "config.yaml"):
    import uvicorn
    config = DreamcatcherConfig.load(config_path)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
