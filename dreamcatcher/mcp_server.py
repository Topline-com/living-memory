"""
Living Memory MCP Server for Claude Code
=========================================
Exposes the Living Memory personal memory system to Claude Code via the
Model Context Protocol (MCP). This is a thin stdio bridge — all memory
logic runs in the separate Living Memory server process.

Integration points:
  instructions → POST /context  (personal memory injected at session start)
  tools        → living_memory_recall, living_memory_status, living_memory_save_session

Requires: a running Living Memory server (default http://localhost:8420).
Start the server: dreamcatcher serve

Usage:
  dreamcatcher mcp                         Start as MCP server (stdio)
  python -m dreamcatcher.mcp_server        Same, direct invocation
"""
import os
import sys
import json
import logging

import httpx

# ── Logging (stderr only — stdout is the MCP JSON-RPC transport) ────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[Living Memory] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────
SERVER_URL = os.environ.get("DREAMCATCHER_SERVER_URL", "http://localhost:8420").rstrip("/")
AGENT_NAME = os.environ.get("DREAMCATCHER_AGENT_NAME", "claude-code")
PREFETCH_TIMEOUT = 5.0
INGEST_TIMEOUT = 10.0
HEALTH_TIMEOUT = 3.0

# ── Circuit breaker (matches Hermes/OpenClaw: 5 consecutive failures) ─
_consecutive_failures = 0
_max_failures = 5
_available = False
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=PREFETCH_TIMEOUT)
    return _client


def _circuit_open() -> bool:
    return _consecutive_failures >= _max_failures


def _record_success():
    global _consecutive_failures
    _consecutive_failures = 0


def _record_failure(context: str):
    global _consecutive_failures
    _consecutive_failures += 1
    if _consecutive_failures == _max_failures:
        logger.warning(
            f"Circuit breaker tripped after {_max_failures} consecutive failures "
            f"(last: {context}). Tools will return unavailable until server recovers."
        )
    elif _consecutive_failures < _max_failures:
        logger.debug(f"Failure ({_consecutive_failures}/{_max_failures}): {context}")


# ── Build MCP instructions (fetches personal context on startup) ────

_INSTRUCTIONS_HEADER = """\
You have access to Living Memory, a personal memory system that has been
trained on the user's complete interaction history. The memory model
encodes knowledge about the user's projects, preferences, relationships,
and patterns into its weights through nightly re-fine-tuning.

Available tools:
- living_memory_recall: Query specific memories about the user. Use this
  when you need to recall facts not already in context — for example,
  details about a project mentioned weeks ago, or a preference expressed
  in a different conversation.
- living_memory_status: Check memory model health and statistics.
- living_memory_save_session: Save the current conversation for tonight's
  memory consolidation training run.

IMPORTANT — Session saving:
At the end of every meaningful conversation (one that contains personal
information, project updates, decisions, preferences, or any facts worth
remembering about the user), call living_memory_save_session with the FULL
UNFILTERED conversation transcript. Include every user message and every
assistant response verbatim — do not summarize, truncate, or paraphrase.

The extraction LLM needs the complete raw conversation to produce useful
memories. Brief summaries strip out the specific details (names, decisions,
reasoning, preferences, exact phrasing) that make training pairs valuable.
More data in = better memories out.

Skip saving only for trivial single-command interactions with no personal
content."""


def _build_instructions() -> str:
    """Fetch personal context from Living Memory and assemble MCP instructions."""
    global _available

    instructions = _INSTRUCTIONS_HEADER

    try:
        client = _get_client()
        resp = client.get(f"{SERVER_URL}/health", timeout=HEALTH_TIMEOUT)
        if resp.status_code == 200:
            _available = True
            health = resp.json()
            logger.info(
                f"Connected to Living Memory at {SERVER_URL} "
                f"(model_loaded={health.get('model_loaded', False)})"
            )
        else:
            logger.warning(f"Living Memory returned {resp.status_code}")
            return instructions
    except Exception as e:
        logger.warning(
            f"Living Memory server not reachable at {SERVER_URL}: {e}. "
            f"Start it with: dreamcatcher serve"
        )
        return instructions

    # Fetch personal context
    try:
        resp = client.post(
            f"{SERVER_URL}/context",
            json={
                "query": "comprehensive user profile and current context",
                "agent_name": AGENT_NAME,
                "max_tokens": 1024,
            },
            timeout=PREFETCH_TIMEOUT,
        )
        resp.raise_for_status()
        context = resp.json().get("response", "")
        if context:
            instructions += f"\n\nCurrent personal context from the memory model:\n{context}"
            _record_success()
            logger.info("Personal context loaded into MCP instructions")
    except Exception as e:
        _record_failure(f"context fetch: {e}")
        logger.warning(f"Could not fetch personal context: {e}")

    return instructions


# ── FastMCP server ──────────────────────────────────────────────────

def _create_server():
    """Create and configure the FastMCP server instance."""
    from mcp.server.fastmcp import FastMCP

    instructions = _build_instructions()
    mcp = FastMCP("Living Memory", instructions=instructions)

    # ── Tool: living_memory_recall ─────────────────────────────

    @mcp.tool()
    def living_memory_recall(query: str) -> str:
        """Query the user's personal memory model for specific information.
        Use this when you need to recall facts about the user that aren't
        in the automatic context — details about projects, preferences,
        relationships, or decisions from past conversations."""
        if not _available:
            return (
                "Living Memory server is not available. "
                "Start it with: dreamcatcher serve"
            )
        if _circuit_open():
            return "Living Memory is temporarily unavailable (circuit breaker open)."

        try:
            client = _get_client()
            resp = client.post(
                f"{SERVER_URL}/recall",
                json={"query": query, "max_tokens": 512},
                timeout=PREFETCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            memories = data.get("memories", [])

            if not memories:
                return "No memories found for this query."

            lines = []
            for m in memories:
                cat = m.get("category", "?")
                content = m.get("content", "")
                confidence = m.get("confidence", 0)
                lines.append(f"[{cat}] {content} (confidence: {confidence:.0%})")

            _record_success()
            return "\n".join(lines)

        except Exception as e:
            _record_failure(f"recall: {e}")
            return f"Memory recall failed: {e}"

    # ── Tool: living_memory_status ─────────────────────────────

    @mcp.tool()
    def living_memory_status() -> str:
        """Check the health and statistics of the user's personal memory
        model. Shows model status, memory counts, training history, and
        category breakdown."""
        if not _available:
            return (
                "Living Memory server is not available. "
                "Start it with: dreamcatcher serve"
            )

        try:
            client = _get_client()
            resp = client.get(f"{SERVER_URL}/health", timeout=HEALTH_TIMEOUT)
            resp.raise_for_status()
            health = resp.json()
            stats = health.get("stats", {})

            result = {
                "status": "ok",
                "model_loaded": health.get("model_loaded", False),
                "model_path": health.get("model_path", ""),
                "active_memories": stats.get("active_memories", 0),
                "total_sessions": stats.get("total_sessions", 0),
                "unprocessed_sessions": stats.get("unprocessed_sessions", 0),
                "total_training_examples": stats.get("total_training_examples", 0),
                "training_runs": stats.get("training_runs", 0),
                "memories_by_category": stats.get("memories_by_category", {}),
            }

            _record_success()
            return json.dumps(result, indent=2)

        except Exception as e:
            _record_failure(f"status: {e}")
            return f"Status check failed: {e}"

    # ── Tool: living_memory_save_session ───────────────────────

    @mcp.tool()
    def living_memory_save_session(transcript: str) -> str:
        """Save the current conversation transcript for tonight's memory
        consolidation. Call this at the end of every meaningful conversation
        that contains personal information, project updates, decisions, or
        preferences.

        IMPORTANT: Save the FULL UNFILTERED transcript — every user message
        and every assistant response, verbatim. Do not summarize, truncate,
        or paraphrase. The extraction LLM needs the complete raw conversation
        to produce high-quality memories. More data in = better memories out."""
        if not transcript or not transcript.strip():
            return "Nothing to save — transcript is empty."

        if not _available:
            return (
                "Living Memory server is not available. "
                "Transcript was NOT saved. Start the server with: dreamcatcher serve"
            )

        try:
            client = _get_client()
            resp = client.post(
                f"{SERVER_URL}/ingest",
                json={
                    "transcript": transcript,
                    "agent_name": AGENT_NAME,
                },
                timeout=INGEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            session_id = data.get("session_id", "?")

            _record_success()
            logger.info(f"Session saved: {session_id} ({len(transcript)} chars)")
            return (
                f"Session saved for tonight's memory consolidation.\n"
                f"  Session ID: {session_id}\n"
                f"  Characters: {len(transcript)}\n"
                f"  Status: {data.get('status', 'stored')}"
            )

        except Exception as e:
            _record_failure(f"ingest: {e}")
            return f"Failed to save session: {e}"

    return mcp


# ── Entry point ─────────────────────────────────────────────────────

def main():
    """Run the Living Memory MCP server over stdio."""
    try:
        mcp = _create_server()
        mcp.run(transport="stdio")
    except ImportError:
        print(
            "Error: MCP package not installed.\n"
            "Install it with: pip install dreamcatcher-memory[claude-code]",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
