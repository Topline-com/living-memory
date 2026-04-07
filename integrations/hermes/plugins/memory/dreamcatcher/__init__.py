"""
Dreamcatcher Memory Provider for Hermes Agent
==============================================
Parametric memory layer that internalizes personal knowledge into a
compact model's weights through nightly re-fine-tuning, then serves
structured context via a local HTTP API.

Unlike retrieval-based providers (Mem0, Hindsight, RetainDB), Dreamcatcher
encodes knowledge parametrically — the memory model genuinely "knows" the
user rather than looking them up. The plugin itself is a thin HTTP client;
all memory logic runs in the separate Dreamcatcher server process.

Integration points:
  prefetch   → GET /context  (personal memory injected every turn)
  sync_turn  → accumulates transcript in memory
  session_end → POST /ingest (transcript saved for tonight's training)
  tools      → dreamcatcher_recall, dreamcatcher_status

Requires: a running Dreamcatcher server (default http://localhost:8420).
Install Dreamcatcher separately: pip install dreamcatcher-memory
Start the server: dreamcatcher serve
"""
import os
import json
import threading
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Provider metadata ─────────────────────────────────────────────
PROVIDER_NAME = "dreamcatcher"
PROVIDER_DISPLAY_NAME = "Dreamcatcher"
PROVIDER_DESCRIPTION = (
    "Parametric memory — a compact model re-fine-tuned nightly on your "
    "complete personal data. Knowledge lives in weights, not retrieval."
)

# ── Default configuration ─────────────────────────────────────────
_DEFAULT_SERVER_URL = "http://localhost:8420"
_DEFAULT_AGENT_NAME = "hermes"
_PREFETCH_TIMEOUT = 5.0   # seconds — prefetch must be fast
_INGEST_TIMEOUT = 10.0    # seconds — ingest can be slightly slower


class DreamcatcherProvider:
    """
    Hermes MemoryProvider implementation for the Dreamcatcher architecture.

    This plugin is deliberately simple — it's an HTTP client that talks to
    the local Dreamcatcher server. All memory extraction, training, model
    management, compression, and linting happen in the Dreamcatcher process,
    not here. The plugin's job is to bridge two interfaces:

      Hermes lifecycle hooks  ←→  Dreamcatcher HTTP API
    """

    # ── Initialization ────────────────────────────────────────────

    def __init__(self):
        self._server_url = _DEFAULT_SERVER_URL
        self._agent_name = _DEFAULT_AGENT_NAME
        self._session_id = None
        self._hermes_home = None
        self._httpx = None          # Lazy-loaded httpx client
        self._client_lock = threading.Lock()

        # Transcript accumulator — built up turn by turn, flushed at session end
        self._transcript_lines = []
        self._transcript_lock = threading.Lock()

        # Background thread tracking
        self._prefetch_thread = None
        self._ingest_thread = None
        self._last_context = ""     # Cached prefetch result for on_pre_compress

        # Health tracking
        self._available = None      # None = not checked yet
        self._consecutive_failures = 0
        self._max_failures = 5      # Circuit breaker threshold

    def initialize(self, **kwargs) -> None:
        """
        Called once at session start. Loads config, verifies the Dreamcatcher
        server is reachable, and stores session context for transcript tagging.
        """
        self._hermes_home = kwargs.get("hermes_home", os.environ.get("HERMES_HOME", ""))
        self._session_id = kwargs.get("session_id", "unknown")

        # Load config from environment or Hermes config
        self._server_url = os.environ.get(
            "DREAMCATCHER_SERVER_URL",
            _DEFAULT_SERVER_URL,
        )
        self._agent_name = os.environ.get(
            "DREAMCATCHER_AGENT_NAME",
            kwargs.get("agent_identity", _DEFAULT_AGENT_NAME),
        )

        # Verify server is reachable (non-blocking — sets self._available)
        try:
            client = self._get_client()
            resp = client.get(f"{self._server_url}/health", timeout=3.0)
            if resp.status_code == 200:
                health = resp.json()
                model_loaded = health.get("model_loaded", False)
                self._available = True
                logger.info(
                    f"Dreamcatcher connected at {self._server_url} "
                    f"(model_loaded={model_loaded})"
                )
            else:
                self._available = False
                logger.warning(
                    f"Dreamcatcher server returned {resp.status_code} — "
                    f"running without parametric memory"
                )
        except Exception as e:
            self._available = False
            logger.warning(
                f"Dreamcatcher server not reachable at {self._server_url}: {e}. "
                f"Start it with: dreamcatcher serve"
            )

    # ── Availability ──────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Returns True if the Dreamcatcher server was reachable at init time.
        Does NOT make a network call — just returns the cached check result.
        This matches the ABC contract (is_available should be fast).
        """
        return self._available is True

    # ── Prefetch (the core integration point) ─────────────────────

    def prefetch(self, query: str = "", session_id: str = "", **kwargs) -> str:
        """
        Called before each turn to get personal context for injection into
        the agent's system prompt. Hits Dreamcatcher's /context endpoint
        which runs inference on the trained memory model (~50ms locally).

        Returns a structured <personal_memory> block ready for prompt injection.
        """
        if not self._available or self._circuit_open():
            return ""

        try:
            client = self._get_client()
            resp = client.post(
                f"{self._server_url}/context",
                json={
                    "query": query or "general context",
                    "agent_name": self._agent_name,
                },
                timeout=_PREFETCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            context = data.get("response", "")
            self._last_context = context  # Cache for on_pre_compress
            self._record_success()
            return context

        except Exception as e:
            self._record_failure(f"prefetch: {e}")
            return self._last_context  # Return stale context rather than nothing

    def queue_prefetch(self, query: str = "", session_id: str = "", **kwargs) -> None:
        """
        Non-blocking prefetch — runs in a background thread so the agent
        doesn't wait for memory retrieval before starting the next turn.
        """
        if not self._available:
            return

        # Wait for any previous prefetch to complete before starting a new one
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=_PREFETCH_TIMEOUT)

        self._prefetch_thread = threading.Thread(
            target=self.prefetch,
            args=(query,),
            kwargs={"session_id": session_id},
            daemon=True,
        )
        self._prefetch_thread.start()

    # ── Turn sync (transcript accumulation) ───────────────────────

    def sync_turn(self, user_msg: str = "", assistant_msg: str = "",
                  session_id: str = "", **kwargs) -> None:
        """
        Called after each completed turn. Accumulates the conversation
        transcript in memory — does NOT send to the server yet. The full
        transcript is flushed at session end via on_session_end().

        This is non-blocking and thread-safe.
        """
        with self._transcript_lock:
            if user_msg:
                self._transcript_lines.append(f"User: {user_msg}")
            if assistant_msg:
                # Save full content — truncation strips details that extraction needs
                self._transcript_lines.append(f"Assistant: {assistant_msg}")

    # ── Session lifecycle ─────────────────────────────────────────

    def on_turn_start(self, **kwargs) -> None:
        """Called at the start of each turn. No-op for Dreamcatcher."""
        pass

    def on_session_end(self, session_id: str = "", **kwargs) -> None:
        """
        Called at actual session boundary (not after every turn — that bug
        was fixed in the review). Flushes the accumulated transcript to
        Dreamcatcher's /ingest endpoint for tonight's extraction pipeline.
        """
        if not self._available:
            return

        with self._transcript_lock:
            if not self._transcript_lines:
                return
            transcript = "\n\n".join(self._transcript_lines)
            self._transcript_lines = []  # Clear after capturing

        # Ingest in a background thread so shutdown isn't blocked
        self._ingest_thread = threading.Thread(
            target=self._ingest_transcript,
            args=(transcript,),
            daemon=True,
        )
        self._ingest_thread.start()

    def _ingest_transcript(self, transcript: str) -> None:
        """POST the full session transcript to Dreamcatcher for tonight's training."""
        try:
            client = self._get_client()
            resp = client.post(
                f"{self._server_url}/ingest",
                json={
                    "transcript": transcript,
                    "agent_name": self._agent_name,
                },
                timeout=_INGEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            session_id = data.get("session_id", "?")
            logger.info(f"Dreamcatcher ingested session {session_id} ({len(transcript)} chars)")
            self._record_success()
        except Exception as e:
            self._record_failure(f"ingest: {e}")

    # ── Memory write bridge ───────────────────────────────────────

    def on_memory_write(self, content: str = "", **kwargs) -> None:
        """
        Called when Hermes's built-in memory (MEMORY.md/USER.md) writes a fact.
        We append it to the transcript so Dreamcatcher can extract it tonight.
        This captures user corrections and explicit memory instructions that
        might not appear in the conversational transcript.
        """
        if content:
            with self._transcript_lock:
                self._transcript_lines.append(
                    f"[Memory write] {content}"
                )

    # ── Context compression ───────────────────────────────────────

    def on_pre_compress(self, **kwargs) -> str:
        """
        Called before Hermes compresses the context window. Returns a string
        that gets preserved in the compression summary — ensuring personal
        memory context survives compression.
        """
        if self._last_context:
            return (
                f"[Dreamcatcher personal memory was active this session. "
                f"Key context preserved from parametric memory model.]"
            )
        return ""

    # ── Delegation ────────────────────────────────────────────────

    def on_delegation(self, task: str = "", result: str = "", **kwargs) -> None:
        """
        Called when a subagent completes a delegated task. We capture this
        in the transcript so Dreamcatcher learns about the full agent
        workflow — what was delegated and what came back.
        """
        if task or result:
            with self._transcript_lock:
                self._transcript_lines.append(
                    f"[Delegation] Task: {task[:500]}\nResult: {(result or '')[:1000]}"
                )

    # ── Shutdown ──────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean up: flush any remaining transcript, join background threads."""
        # Flush transcript if session_end wasn't called
        with self._transcript_lock:
            if self._transcript_lines:
                transcript = "\n\n".join(self._transcript_lines)
                self._transcript_lines = []
                if self._available:
                    self._ingest_transcript(transcript)

        # Wait for background threads to complete
        for thread in (self._prefetch_thread, self._ingest_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)

    # ── Tool schemas ──────────────────────────────────────────────

    def get_tool_schemas(self) -> list:
        """
        Register tools that the agent can call explicitly. Automatic context
        injection happens via prefetch() — these tools are for when the agent
        wants to make an explicit, targeted memory query.
        """
        return [
            {
                "name": "dreamcatcher_recall",
                "description": (
                    "Query the user's parametric memory model for specific information. "
                    "Use this when you need to recall specific facts about the user that "
                    "weren't in the automatic context injection — for example, details "
                    "about a project they mentioned weeks ago, or a preference they "
                    "expressed in a different context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to recall about the user",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "dreamcatcher_status",
                "description": (
                    "Check the health and status of the user's memory model. "
                    "Shows when the model was last trained, how many memories "
                    "it contains, and whether the nightly pipeline is healthy."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, tool_input: dict, **kwargs) -> str:
        """Route explicit tool calls to the appropriate Dreamcatcher endpoint."""
        if tool_name == "dreamcatcher_recall":
            return self._tool_recall(tool_input.get("query", ""))
        elif tool_name == "dreamcatcher_status":
            return self._tool_status()
        return f"Unknown tool: {tool_name}"

    def _tool_recall(self, query: str) -> str:
        """Explicit memory recall — hits the /recall endpoint for targeted queries."""
        if not self._available:
            return "Dreamcatcher server is not available."
        try:
            client = self._get_client()
            resp = client.post(
                f"{self._server_url}/recall",
                json={"query": query},
                timeout=_PREFETCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            memories = data.get("memories", [])
            if not memories:
                return "No memories found for this query."
            lines = []
            for m in memories:
                lines.append(f"[{m.get('category', '?')}] {m.get('content', '')}")
            return "\n".join(lines)
        except Exception as e:
            return f"Memory recall failed: {e}"

    def _tool_status(self) -> str:
        """Check Dreamcatcher server health and memory statistics."""
        if not self._available:
            return "Dreamcatcher server is not available."
        try:
            client = self._get_client()
            resp = client.get(f"{self._server_url}/health", timeout=3.0)
            resp.raise_for_status()
            health = resp.json()
            stats = health.get("stats", {})
            return json.dumps({
                "status": "ok",
                "model_loaded": health.get("model_loaded", False),
                "active_memories": stats.get("active_memories", 0),
                "total_training_examples": stats.get("total_training_examples", 0),
                "training_runs": stats.get("training_runs", 0),
                "memories_by_category": stats.get("memories_by_category", {}),
            }, indent=2)
        except Exception as e:
            return f"Status check failed: {e}"

    # ── System prompt block ───────────────────────────────────────

    @property
    def system_prompt_block(self) -> str:
        """
        Injected into the agent's system prompt alongside MEMORY.md/USER.md.
        Tells the agent about Dreamcatcher's automatic context injection and
        the explicit recall tool.
        """
        if not self._available:
            return ""
        return (
            "## Parametric Memory (Dreamcatcher)\n"
            "You have access to a personal memory model that has been trained on "
            "the user's complete interaction history. Personal context about the "
            "user's projects, preferences, relationships, and patterns is "
            "automatically injected at the start of each turn — you don't need "
            "to request it. For specific queries about things not in the automatic "
            "context, use the dreamcatcher_recall tool.\n"
        )

    # ── Config schema (for hermes memory setup wizard) ────────────

    @staticmethod
    def get_config_schema() -> list:
        """Config fields shown in the interactive setup wizard."""
        return [
            {
                "key": "server_url",
                "label": "Dreamcatcher Server URL",
                "type": "string",
                "default": _DEFAULT_SERVER_URL,
                "env_var": "DREAMCATCHER_SERVER_URL",
                "help": "URL of your running Dreamcatcher server",
            },
            {
                "key": "agent_name",
                "label": "Agent Name",
                "type": "string",
                "default": _DEFAULT_AGENT_NAME,
                "env_var": "DREAMCATCHER_AGENT_NAME",
                "help": "Name tag for this agent's transcripts (used in training)",
            },
        ]

    @staticmethod
    def save_config(values: dict, hermes_home: str = "") -> None:
        """
        Dreamcatcher config is env-var-only (server_url and agent_name).
        The setup wizard writes these to .env. No separate config file needed.
        """
        pass  # Env vars handled by the wizard's standard .env writer

    # ── HTTP client (lazy, thread-safe) ───────────────────────────

    def _get_client(self):
        """Lazy-initialize the httpx client with thread safety."""
        if self._httpx is None:
            with self._client_lock:
                if self._httpx is None:
                    import httpx
                    self._httpx = httpx.Client(timeout=_PREFETCH_TIMEOUT)
        return self._httpx

    # ── Circuit breaker ───────────────────────────────────────────

    def _circuit_open(self) -> bool:
        """After N consecutive failures, stop hammering the server."""
        return self._consecutive_failures >= self._max_failures

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self, context: str):
        self._consecutive_failures += 1
        if self._consecutive_failures == self._max_failures:
            logger.warning(
                f"Dreamcatcher circuit breaker tripped after {self._max_failures} "
                f"consecutive failures (last: {context}). Will stop calling until "
                f"next session."
            )
        elif self._consecutive_failures < self._max_failures:
            logger.debug(f"Dreamcatcher failure ({self._consecutive_failures}): {context}")


# ── Plugin registration (called by Hermes plugin discovery) ───────

def register():
    """Entry point for the Hermes plugin system."""
    return DreamcatcherProvider()
