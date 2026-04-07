"""
Dreamcatcher Database — Living Memory Layer
=============================================
SQLite storage for sessions, memories, and training examples.

Key design: training examples carry a `pair_index` indicating their
generality order within a memory (0 = most general, 4 = most specific).
This enables semantic compression: for older memories, only the most
general pairs (pair_index < 2) are included in the nightly training set,
replicating the brain's episodic-to-semantic transition.
"""
import sqlite3
import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class MemoryDB:
    def __init__(self, db_path: str = "./data/memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT,
                    timestamp TEXT NOT NULL,
                    raw_transcript TEXT NOT NULL,
                    token_count INTEGER,
                    processed INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id),
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT,
                    superseded_by TEXT,
                    active INTEGER DEFAULT 1
                );

                -- pair_index: generality order within a source memory.
                --   0 = most general (broadest retrieval surface)
                --   1 = second most general
                --   2-4 = increasingly specific/contextual
                -- Semantic compression keeps only pair_index <= 1 for old memories.
                CREATE TABLE IF NOT EXISTS training_examples (
                    id TEXT PRIMARY KEY,
                    memory_ids TEXT,
                    instruction TEXT NOT NULL,
                    response TEXT NOT NULL,
                    category TEXT,
                    pair_index INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_path TEXT,
                    num_examples INTEGER,
                    num_full_density INTEGER,
                    num_compressed INTEGER,
                    loss_final REAL,
                    duration_seconds REAL,
                    model_name TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_processed ON sessions(processed);
                CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
                CREATE INDEX IF NOT EXISTS idx_training_created ON training_examples(created_at);
            """)
            # Migrations for existing databases
            try:
                conn.execute("SELECT pair_index FROM training_examples LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE training_examples ADD COLUMN pair_index INTEGER DEFAULT 0")
            try:
                conn.execute("SELECT last_seen_at FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN last_seen_at TEXT")

    # ── Sessions ────────────────────────────────────────────────

    def add_session(self, transcript: str, agent_name: str = "unknown",
                    session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = hashlib.sha256(
                f"{datetime.now(timezone.utc).isoformat()}:{transcript[:200]}".encode()
            ).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, agent_name, timestamp, raw_transcript, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, agent_name, now, transcript, int(len(transcript.split()) * 1.3))
            )
        return session_id

    def get_unprocessed_sessions(self) -> list[dict]:
        """Only returns sessions not yet sent to the frontier LLM for extraction."""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM sessions WHERE processed = 0 ORDER BY timestamp"
            ).fetchall()]

    def mark_session_processed(self, session_id: str):
        with self._conn() as conn:
            conn.execute("UPDATE sessions SET processed = 1 WHERE id = ?", (session_id,))

    # ── Memories ────────────────────────────────────────────────

    def add_memory(self, content: str, category: str, session_id: str,
                   confidence: float = 1.0) -> str:
        memory_id = hashlib.sha256(f"{category}:{content}".encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            # INSERT OR IGNORE preserves original session_id/created_at provenance.
            # On re-encounter, only last_seen_at and confidence are refreshed.
            conn.execute(
                "INSERT OR IGNORE INTO memories (id, session_id, category, content, confidence, created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, session_id, category, content, confidence, now, now)
            )
            conn.execute(
                "UPDATE memories SET last_seen_at = ?, confidence = MAX(confidence, ?) WHERE id = ?",
                (now, confidence, memory_id)
            )
        return memory_id

    def get_active_memories(self, category: Optional[str] = None, limit: int = 500) -> list[dict]:
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE active = 1 AND category = ? ORDER BY created_at DESC LIMIT ?",
                    (category, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE active = 1 ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
            return [dict(r) for r in rows]

    def supersede_memory(self, old_id: str, new_id: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET active = 0, superseded_by = ? WHERE id = ?", (new_id, old_id))

    # ── Training Examples ───────────────────────────────────────

    def add_training_example(self, instruction: str, response: str,
                             category: str, memory_ids: list[str] = None,
                             pair_index: int = 0) -> str:
        """
        Store a training Q&A pair. pair_index indicates generality:
        0 = most general (kept during compression), 4 = most specific.
        """
        example_id = hashlib.sha256(f"{instruction}:{response}".encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            # INSERT OR IGNORE preserves original created_at provenance.
            conn.execute(
                "INSERT OR IGNORE INTO training_examples "
                "(id, memory_ids, instruction, response, category, pair_index, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (example_id, json.dumps(memory_ids or []), instruction, response,
                 category, pair_index, now))
        return example_id

    def get_all_training_examples(self) -> list[dict]:
        """Return ALL training examples (no compression)."""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM training_examples ORDER BY created_at").fetchall()]

    def get_training_set_with_compression(self, compression_age_days: int = 180,
                                          max_pair_index_old: int = 1) -> dict:
        """
        Build the nightly training set with semantic compression.

        Recent examples (< compression_age_days): ALL pairs included.
        Old examples (>= compression_age_days): only pair_index <= max_pair_index_old.

        Organically reinforced memories (important facts discussed across
        many sessions) naturally accumulate RECENT training pairs that
        bypass compression entirely. Compression primarily affects
        one-off memories — exactly those that should get leaner encoding.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=compression_age_days)).isoformat()
        with self._conn() as conn:
            # Build sets to filter training examples whose source memories were superseded
            active_ids = {row[0] for row in conn.execute(
                "SELECT id FROM memories WHERE active = 1").fetchall()}
            inactive_ids = {row[0] for row in conn.execute(
                "SELECT id FROM memories WHERE active = 0").fetchall()}

            def _has_active_source(example: dict) -> bool:
                """Exclude example only if ALL its source memories are known-inactive."""
                ids = json.loads(example.get("memory_ids", "[]"))
                if not ids:
                    return True  # No source tracking — keep
                known = [mid for mid in ids if mid in active_ids or mid in inactive_ids]
                if not known:
                    return True  # IDs don't match any memory — keep (legacy data)
                return any(mid in active_ids for mid in known)

            # Recent: full episodic density
            recent = [dict(r) for r in conn.execute(
                "SELECT * FROM training_examples WHERE created_at >= ? ORDER BY created_at",
                (cutoff,)).fetchall()]
            recent = [e for e in recent if _has_active_source(e)]
            # Old: semantic density only (most general pairs)
            old_included = [dict(r) for r in conn.execute(
                "SELECT * FROM training_examples WHERE created_at < ? AND pair_index <= ? ORDER BY created_at",
                (cutoff, max_pair_index_old)).fetchall()]
            old_included = [e for e in old_included if _has_active_source(e)]
            # Stats
            old_total = conn.execute(
                "SELECT COUNT(*) FROM training_examples WHERE created_at < ?",
                (cutoff,)).fetchone()[0]

        return {
            "examples": recent + old_included,
            "n_full": len(recent),
            "n_compressed": len(old_included),
            "n_dropped": old_total - len(old_included),
        }

    def get_training_example_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]

    # ── Training Runs ───────────────────────────────────────────

    def log_training_run(self, model_path: str, num_examples: int,
                         loss_final: float, duration_seconds: float,
                         model_name: str = "", num_full_density: int = 0,
                         num_compressed: int = 0) -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO training_runs "
                "(id, timestamp, model_path, num_examples, num_full_density, num_compressed, "
                "loss_final, duration_seconds, model_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, now, model_path, num_examples, num_full_density, num_compressed,
                 loss_final, duration_seconds, model_name))
        return run_id

    # ── Stats ───────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._conn() as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
            return {
                "total_sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "unprocessed_sessions": conn.execute("SELECT COUNT(*) FROM sessions WHERE processed = 0").fetchone()[0],
                "active_memories": conn.execute("SELECT COUNT(*) FROM memories WHERE active = 1").fetchone()[0],
                "total_training_examples": conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0],
                "recent_examples": conn.execute("SELECT COUNT(*) FROM training_examples WHERE created_at >= ?", (cutoff,)).fetchone()[0],
                "old_examples": conn.execute("SELECT COUNT(*) FROM training_examples WHERE created_at < ?", (cutoff,)).fetchone()[0],
                "training_runs": conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0],
                "memories_by_category": dict(conn.execute(
                    "SELECT category, COUNT(*) FROM memories WHERE active = 1 GROUP BY category").fetchall()),
            }
