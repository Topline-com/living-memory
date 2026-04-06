# Living Memory Integration for Claude Code

Personal memory for Claude Code via the Model Context Protocol (MCP). Your memory model's knowledge is injected into every session, conversations are automatically saved for nightly training, and you can query memories on demand.

## Quick Setup

```bash
pip install dreamcatcher-memory[claude-code]
dreamcatcher setup claude-code --global
```

That's it. Restart Claude Code and your personal memory is active.

## What It Does

| Event | Living Memory Action |
|-------|-------------------|
| Session starts | Fetches personal context from `/context`, injects into MCP instructions |
| Agent calls `living_memory_recall` | Queries memory model via `/recall` |
| Agent calls `living_memory_save_session` | Saves transcript via `/ingest` for tonight's training |
| Tonight at 3 AM | Nightly pipeline extracts, trains, deploys updated model |

## Tools

| Tool | Description |
|------|-------------|
| `living_memory_recall` | Query specific memories (projects, preferences, relationships) |
| `living_memory_status` | Check model health, memory counts, training history |
| `living_memory_save_session` | Save the conversation transcript for nightly consolidation |

## Architecture

```
┌──────────────┐    stdio/JSON-RPC    ┌─────────────────┐    HTTP    ┌───────────────────────┐
│  Claude Code  │ ◄──────────────────► │  mcp_server.py  │ ◄────────► │  Living Memory Server │
│  (MCP client) │                      │  (MCP bridge)   │           │  (FastAPI :8420)      │
└──────────────┘                      └─────────────────┘           └──────────┬────────────┘
                                                                               │
                                                                    ┌──────────▼──────────┐
                                                                    │  Memory Model (local)│
                                                                    │  + SQLite ledger     │
                                                                    └─────────────────────┘
```

The MCP server is a thin bridge — all memory logic (extraction, training, inference, compression) runs in the Living Memory server process.

## Manual Setup

If you prefer to configure manually instead of using `dreamcatcher setup`:

### 1. Install

```bash
pip install dreamcatcher-memory[claude-code]
```

### 2. Configure Claude Code

Add to `~/.claude/settings.json` (global) or `.claude/settings.json` (project):

```json
{
  "mcpServers": {
    "living-memory": {
      "type": "stdio",
      "command": "dreamcatcher",
      "args": ["mcp"],
      "env": {
        "DREAMCATCHER_SERVER_URL": "http://localhost:8420"
      }
    }
  }
}
```

### 3. Start the Dreamcatcher server

```bash
dreamcatcher serve
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DREAMCATCHER_SERVER_URL` | `http://localhost:8420` | Dreamcatcher server URL |
| `DREAMCATCHER_AGENT_NAME` | `claude-code` | Agent name tag for transcripts |

## Setup Command Options

```bash
dreamcatcher setup claude-code [options]

Options:
  --global      Write to ~/.claude/settings.json (default: project .claude/)
  --url URL     Override Dreamcatcher server URL
  --claude-md   Also generate a CLAUDE.md file in the current directory
```

## How Auto-Capture Works

Unlike the Hermes and OpenClaw integrations (which hook into agent lifecycle events), MCP does not expose per-turn or session-end callbacks. Instead, the MCP server's `instructions` field tells Claude to call `living_memory_save_session` at the end of every meaningful conversation. This is reliable in practice because Claude follows MCP tool-use instructions consistently.

## Troubleshooting

**"Dreamcatcher server is not available"**
- Start the server: `dreamcatcher serve`
- Check it's running: `curl http://localhost:8420/health`

**"model_loaded: false"**
- Run the pipeline first: `dreamcatcher nightly`
- Or ingest some data: `dreamcatcher ingest <transcript-file>`

**"MCP package not installed"**
- Install: `pip install dreamcatcher-memory[claude-code]`

**Context seems stale or empty**
- Check when the model was last trained: `dreamcatcher stats`
- Run a fresh nightly: `dreamcatcher nightly`
- The server has disaster recovery — if the model is >36h stale, it injects recent DB memories directly

**`ModuleNotFoundError: No module named 'dreamcatcher'` / MCP disconnects on restart**
- This is a known issue with editable installs (`pip install -e .`) on uv-managed Python 3.12. The MCP server runs from an arbitrary working directory and can't find the package.
- Fix: `pip install .` (non-editable) or `pip install dreamcatcher-memory[claude-code]`

**Tools don't appear in Claude Code**
- Restart Claude Code after setup
- Verify settings: check `~/.claude/settings.json` or `.claude/settings.json`
- Check MCP logs in Claude Code's developer console
