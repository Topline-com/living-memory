"""
Dreamcatcher Session Collector & Memory Extractor
==============================================
Extracts structured memories from agent session transcripts using a
frontier LLM API. This is the one cloud touchpoint in the architecture:
raw transcripts are sent to the API for extraction, and the resulting
canonical facts + training pairs are stored permanently in SQLite.

The extraction prompt uses a structured schema with core_fact and
explicitly keyed training pairs (semantic → contextual → specific)
to enforce generality ordering. Session dates are injected into the
prompt so that temporal supersession is handled linguistically.

The TrainingDataBuilder renders the nightly training set from the
canonical ledger, applying semantic compression as a rendering policy
(not a deletion operation) for memories older than the configured
compression age threshold.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .database import MemoryDB
from .config import DreamcatcherConfig

# ── Extraction prompt (runs on Claude Sonnet, same as v1) ───────────

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system for the Dreamcatcher architecture. Read the conversation transcript and extract structured memories about the USER.

Categories: preference, fact, project, pattern, relationship, decision

For EACH memory, you MUST:
1. First articulate the core_fact: a single canonical statement of the essential truth.
2. Then generate training pairs in EXPLICIT generality order using keyed names:
   - "semantic": the broadest, most general question that would retrieve this fact
   - "contextual": a question about the broader context this fact belongs to
   - "specific_1" through "specific_3": increasingly narrow questions about details
3. When the user describes a CHANGE from a previous state, embed the session date
   into the core_fact and training pairs to establish the timeline
   (e.g., "As of [date], the user switched from X to Y").

Respond ONLY with a JSON array:
[
  {
    "category": "project",
    "core_fact": "Nowell Creek is a 213-unit multifamily workforce housing project in Berkeley County with HUD 221(d)(4) financing",
    "confidence": 0.95,
    "training_pairs": {
      "semantic": {
        "instruction": "What real estate development projects is the user working on?",
        "response": {"memories": [{"category": "project", "content": "The user is actively developing Nowell Creek, a multifamily workforce housing project in Berkeley County", "confidence": 0.95}]}
      },
      "contextual": {
        "instruction": "What is the user's affordable housing work?",
        "response": {"memories": [{"category": "project", "content": "The user is developing Nowell Creek, a 213-unit workforce housing project at 60% AMI with HUD 221(d)(4) financing and 4% LIHTC", "confidence": 0.95}]}
      },
      "specific_1": {
        "instruction": "What is Nowell Creek?",
        "response": {"memories": [{"category": "project", "content": "Nowell Creek is a 213-unit multifamily workforce housing development in the Cainhoy/Clements Ferry Road corridor of Berkeley County", "confidence": 0.95}]}
      },
      "specific_2": {
        "instruction": "How many units is Nowell Creek?",
        "response": {"memories": [{"category": "fact", "content": "Nowell Creek total is 213 units: Phase 1 = 177 units, Phase 2a = 13 units, Phase 2b = 23 units", "confidence": 0.95}]}
      },
      "specific_3": {
        "instruction": "What financing is the user using for Nowell Creek?",
        "response": {"memories": [{"category": "fact", "content": "Nowell Creek uses HUD 221(d)(4) financing with 4% LIHTC tax-exempt bonds via SC Housing QAP", "confidence": 0.95}]}
      }
    }
  }
]

Rules:
- The "semantic" pair MUST be the broadest possible question that retrieves this fact
- The key ordering (semantic → contextual → specific) is critical for compression
- Training responses MUST be JSON with a "memories" array
- Each response memory has: category, content, confidence
- Include exact numbers, names, and dates — precision matters
- Write content from third person ("The user..." or "Alex...")
- When facts represent changes, include temporal markers with the session date"""

EXTRACTION_USER_TEMPLATE = """You are extracting canonical memory from a session dated {timestamp}.
If the user describes a change from a previous state, embed this date into the
core_fact and training pairs to establish the timeline.

Session transcript:

<session agent="{agent_name}" date="{timestamp}">
{transcript}
</session>

Extract all user memories with core_fact and keyed training pairs. JSON array only."""


class SessionCollector:
    """Ingests sessions and extracts structured memories."""

    def __init__(self, config: DreamcatcherConfig = None):
        self.config = config or DreamcatcherConfig.load()
        self.db = MemoryDB(self.config.db_path)

    def ingest_file(self, filepath: str, agent_name: str = "unknown") -> str:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        return self.ingest_text(path.read_text(encoding="utf-8"), agent_name)

    def ingest_text(self, transcript: str, agent_name: str = "unknown") -> str:
        session_id = self.db.add_session(transcript, agent_name)
        print(f"  Stored session {session_id} ({len(transcript)} chars)")
        return session_id

    def ingest_directory(self, dirpath: str = None, agent_name: str = "unknown") -> list[str]:
        dirpath = dirpath or self.config.sessions_dir
        ids = []
        for pattern in ("*.txt", "*.md", "*.json"):
            for f in sorted(Path(dirpath).glob(pattern)):
                ids.append(self.ingest_file(str(f), agent_name))
        return ids

    async def extract_memories(self, session_id: Optional[str] = None) -> list[dict]:
        """Extract memories from unprocessed sessions via LLM API."""
        if session_id:
            with self.db._conn() as conn:
                row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
                sessions = [dict(row)] if row else []
        else:
            sessions = self.db.get_unprocessed_sessions()

        if not sessions:
            print("  No unprocessed sessions.")
            return []

        all_memories = []
        for session in sessions:
            print(f"  Extracting from session {session['id']}...")
            memories = await self._call_extraction_llm(session)
            all_memories.extend(memories)

            # Store each memory and its training pairs
            for mem in memories:
                # Use core_fact as the memory content if available
                content = mem.get("core_fact", mem.get("content", ""))
                memory_id = self.db.add_memory(
                    content=content,
                    category=mem["category"],
                    session_id=session["id"],
                    confidence=mem.get("confidence", 1.0),
                )

                # Store training pairs with pair_index for semantic compression.
                # The structured schema uses keyed pairs (dict) in generality order.
                # Also supports legacy list format for backward compatibility.
                training_pairs = mem.get("training_pairs", {})

                if isinstance(training_pairs, dict):
                    # New structured format: keys define generality order
                    KEY_ORDER = ["semantic", "contextual", "specific_1", "specific_2", "specific_3"]
                    ordered_pairs = []
                    for key in KEY_ORDER:
                        if key in training_pairs:
                            ordered_pairs.append(training_pairs[key])
                    # Also catch any extra keys not in standard order
                    for key, pair in training_pairs.items():
                        if key not in KEY_ORDER:
                            ordered_pairs.append(pair)
                elif isinstance(training_pairs, list):
                    # Legacy list format: assume most-general-first ordering
                    ordered_pairs = training_pairs
                else:
                    ordered_pairs = []

                for pair_idx, pair in enumerate(ordered_pairs):
                    response = pair["response"]
                    if isinstance(response, dict):
                        response = json.dumps(response)
                    self.db.add_training_example(
                        instruction=pair["instruction"],
                        response=response,
                        category=mem["category"],
                        memory_ids=[memory_id],
                        pair_index=pair_idx,  # 0 = semantic (broadest), kept during compression
                    )

            if memories is None:
                print(f"    → extraction failed, session will be retried next run")
            elif memories:
                self.db.mark_session_processed(session["id"])
                n_pairs = sum(
                    len(m.get("training_pairs", {})) if isinstance(m.get("training_pairs", {}), dict)
                    else len(m.get("training_pairs", []))
                    for m in memories
                )
                print(f"    → {len(memories)} memories, {n_pairs} training pairs")
            else:
                self.db.mark_session_processed(session["id"])
                print(f"    → no memories to extract, session marked done")

        return all_memories

    async def _call_extraction_llm(self, session: dict) -> list[dict]:
        user_msg = EXTRACTION_USER_TEMPLATE.format(
            agent_name=session.get("agent_name", "unknown"),
            timestamp=session.get("timestamp", "unknown"),
            transcript=session["raw_transcript"][:50000],
        )
        provider = self.config.extraction.provider if hasattr(self.config, 'extraction') else "anthropic"
        provider = os.environ.get("DREAMCATCHER_PROVIDER", provider)
        try:
            if provider == "openrouter":
                return await self._call_openrouter(user_msg)
            if provider == "openai":
                return await self._call_openai(user_msg)
            return await self._call_anthropic(user_msg)
        except Exception as e:
            print(f"    ERROR: {e}")
            return None  # None = failure (retry); [] = successful empty (no memories worth extracting)

    async def _call_anthropic(self, user_msg: str) -> list[dict]:
        import anthropic
        model = self.config.extraction.model if hasattr(self.config, 'extraction') else "claude-sonnet-4-20250514"
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=8192,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)

    async def _call_openrouter(self, user_msg: str) -> list[dict]:
        from openai import AsyncOpenAI
        model = self.config.extraction.model if hasattr(self.config, 'extraction') else "anthropic/claude-sonnet-4-5"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = await client.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        if isinstance(result, dict) and "memories" in result:
            return result["memories"]
        return result if isinstance(result, list) else []

    async def _call_openai(self, user_msg: str) -> list[dict]:
        from openai import AsyncOpenAI
        model = self.config.extraction.model if hasattr(self.config, 'extraction') else "gpt-4o-mini"
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        result = json.loads(text)
        if isinstance(result, dict) and "memories" in result:
            return result["memories"]
        return result if isinstance(result, list) else []


class TrainingDataBuilder:
    """
    Builds the nightly training dataset with semantic compression.

    Two mechanisms manage dataset composition:

    1. ORGANIC REINFORCEMENT (automatic, no engineering):
       Important facts naturally accumulate more training pairs over
       time through real-world reuse across multiple sessions. A child's
       allergy discussed in 6 sessions has ~28 pairs; a contractor's
       name from one session has 4. The dataset's composition IS the
       priority signal.

    2. SEMANTIC COMPRESSION (deliberate, age-based):
       Training pairs older than 6 months are filtered to keep only the
       most general questions (pair_index <= 1). This replicates the
       brain's episodic-to-semantic transition: vivid multi-contextual
       encoding consolidates into lean factual encoding over time.
       Reclaims ~60% of historical compute overhead.

    Together, these ensure important knowledge stays densely encoded
    (via organic reinforcement) while the overall dataset stays lean
    (via semantic compression on unimportant old memories).
    """

    SYSTEM_MSG = (
        "You are a personal memory retrieval system. Given a question about the user, "
        "respond with a JSON object containing relevant memories. Each memory has a "
        "category, content, and confidence score. Be precise and factual."
    )

    # Age threshold for compression (days). Memories older than this
    # are reduced from full density (4-5 pairs) to semantic density (1-2 pairs).
    COMPRESSION_AGE_DAYS = 180

    # Maximum pair_index to keep for old memories.
    # 0 = keep only the single most general pair
    # 1 = keep the two most general pairs (recommended)
    MAX_PAIR_INDEX_OLD = 1

    def __init__(self, config: DreamcatcherConfig = None):
        self.config = config or DreamcatcherConfig.load()
        self.db = MemoryDB(self.config.db_path)

    def build_training_set(self) -> list[dict]:
        """
        Build tonight's training set with semantic compression applied.

        The full pipeline:
        1. Query SQLite for ALL training examples
        2. Apply semantic compression (drop specific pairs for old memories)
        3. Format into chat-template training data
        4. Save as JSONL for the trainer

        NOTE: The frontier LLM API is NEVER called here. This method
        operates entirely on locally-stored data in SQLite. The API was
        called earlier, during the extraction step, on new sessions only.
        """
        result = self.db.get_training_set_with_compression(
            compression_age_days=self.COMPRESSION_AGE_DAYS,
            max_pair_index_old=self.MAX_PAIR_INDEX_OLD,
        )

        examples = result["examples"]
        if not examples:
            print("  No training examples in database.")
            return []

        # Format into chat-template training data
        training_data = []
        for ex in examples:
            training_data.append({
                "messages": [
                    {"role": "system", "content": self.SYSTEM_MSG},
                    {"role": "user", "content": ex["instruction"]},
                    {"role": "assistant", "content": ex["response"]},
                ],
            })

        # Save as JSONL
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = Path(self.config.training_dir) / f"full_dataset_{today}.jsonl"
        with open(output_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        # Report stats
        total_in_db = self.db.get_training_example_count()
        print(f"  Training set built with semantic compression:")
        print(f"    Recent (full density):  {result['n_full']} pairs")
        print(f"    Old (compressed):       {result['n_compressed']} pairs kept")
        print(f"    Old (dropped):          {result['n_dropped']} specific pairs excluded")
        print(f"    Total in dataset:       {len(training_data)} of {total_in_db} ({len(training_data)/max(total_in_db,1)*100:.0f}%)")
        print(f"    Saved to: {output_path}")

        return training_data
