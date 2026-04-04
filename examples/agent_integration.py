"""
Dreamcatcher v2 Agent Integration Examples
=====================================
Three patterns for connecting any AI agent to personal memory.
"""
from dreamcatcher_client import PersonalMemory, enhance_system_prompt, generate_claude_md


# ═══════════════════════════════════════════════════════════════════
# PATTERN 1: OpenClaw / Claude Code Integration
# ═══════════════════════════════════════════════════════════════════
# Generate a CLAUDE.md before each coding session so Claude Code
# starts with full personal context.

def openclaw_startup():
    """Run this at the start of each OpenClaw session."""

    # Generate an up-to-date CLAUDE.md from the memory model
    content = generate_claude_md(output_path="CLAUDE.md")
    if content:
        print(f"Updated CLAUDE.md with {len(content)} chars of personal context")
    else:
        print("Dreamcatcher not available — using existing CLAUDE.md")


def openclaw_shutdown(session_log: str):
    """Run this when an OpenClaw session ends."""

    # Save the session for tonight's training
    with PersonalMemory() as memory:
        sid = memory.save_session(session_log, agent_name="openclaw")
        if sid:
            print(f"Session {sid} saved for nightly training")


# ═══════════════════════════════════════════════════════════════════
# PATTERN 2: Direct Prompt Enhancement (works with any agent)
# ═══════════════════════════════════════════════════════════════════
# Wrap your existing system prompt to automatically inject memory.

def run_agent_with_memory(user_message: str):
    """Example: a simple agent that uses Anthropic API with personal memory."""

    # Your base system prompt
    base_prompt = "You are a helpful business and real estate advisor."

    # Enhance it with personal memory (no-op if Dreamcatcher is down)
    enhanced_prompt = enhance_system_prompt(base_prompt, user_message)

    # Now use enhanced_prompt with your LLM API
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-4-6",
    #     system=enhanced_prompt,
    #     messages=[{"role": "user", "content": user_message}],
    # )

    print(f"Enhanced prompt ({len(enhanced_prompt)} chars):")
    print(enhanced_prompt[:500])


# ═══════════════════════════════════════════════════════════════════
# PATTERN 3: Structured Memory Queries (for custom agent logic)
# ═══════════════════════════════════════════════════════════════════
# When your agent needs to make decisions based on user context.

def smart_agent_routing(user_message: str):
    """Example: route to different tools based on user's memory."""

    with PersonalMemory() as memory:
        if not memory.is_available():
            print("Running without personal memory")
            return

        # Get structured memories relevant to the query
        memories = memory.get_memories(user_message)

        # Use the structured data for routing decisions
        project_memories = [m for m in memories if m.get("category") == "project"]
        preference_memories = [m for m in memories if m.get("category") == "preference"]

        if project_memories:
            print(f"Found {len(project_memories)} relevant project memories:")
            for m in project_memories:
                print(f"  [{m['confidence']:.0%}] {m['content']}")

        if preference_memories:
            print(f"\nUser preferences that apply:")
            for m in preference_memories:
                print(f"  {m['content']}")


# ═══════════════════════════════════════════════════════════════════
# PATTERN 4: Session Tracking Context Manager
# ═══════════════════════════════════════════════════════════════════

class TrackedSession:
    """Auto-tracks conversation turns and saves on exit."""

    def __init__(self, agent_name: str = "agent"):
        self.memory = PersonalMemory()
        self.agent_name = agent_name
        self.turns = []
        self.context = ""

    def __enter__(self):
        # Load personal context at session start
        self.context = self.memory.get_context(agent_name=self.agent_name)
        return self

    def __exit__(self, *args):
        # Save the full transcript for tonight's training
        if self.turns:
            transcript = "\n\n".join(
                f"[{role}]: {content}" for role, content in self.turns
            )
            self.memory.save_session(transcript, self.agent_name)
        self.memory.close()

    def add_turn(self, role: str, content: str):
        self.turns.append((role, content))

    def get_system_prompt(self, base_prompt: str) -> str:
        if self.context:
            return f"{base_prompt}\n\n{self.context}"
        return base_prompt


# Usage:
# with TrackedSession("hermes") as session:
#     system = session.get_system_prompt("You are a helpful agent.")
#     session.add_turn("user", "What's the status of Nowell Creek?")
#     # ... get response from Opus ...
#     session.add_turn("assistant", response)


if __name__ == "__main__":
    print("=" * 60)
    print("  Dreamcatcher v2 Integration Examples")
    print("=" * 60)
    print()
    print("Pattern 1: OpenClaw — generate CLAUDE.md at startup")
    print("Pattern 2: Prompt Enhancement — one-line wrapper")
    print("Pattern 3: Structured Queries — access raw memory data")
    print("Pattern 4: Tracked Sessions — auto-save transcripts")
    print()
    print("Run with Dreamcatcher server active to see live output.")
    print("Start server: python -m dreamcatcher serve")
